# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
import topi
from ..nn.dense import dense, dense_default
from .. import tag
from .. import generic

from ..util import traverse_inline, get_const_tuple
from tvm.autotvm.task.space import SplitEntity

@autotvm.register_topi_compute(dense, "snps_ev", "direct")
def dense_snps_ev(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for snps_ev backend.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    target = tvm.target.current_target()
    return dense_default(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule(generic.schedule_dense, "snps_ev", "direct")
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    target = tvm.target.current_target()

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs

    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(C):
        A, _ = C.op.input_tensors
        batch, _ = get_const_tuple(A.shape)
        if batch < 32:
            return schedule_dense_small_batch(cfg, s, C)
        return schedule_dense_large_batch(cfg, s, C)

    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_dense_small_batch(cfg, s, C):
    """Schedule float32/64 dense with small batch size"""
    A, _ = C.op.input_tensors
    _, in_dim = get_const_tuple(A.shape)
    cfg.define_split('tile_k', in_dim, num_outputs=2)
    if cfg.is_fallback:
        cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])

    _, kf = cfg['tile_k'].apply(s, C, C.op.reduce_axis[0])
    CF = s.rfactor(C, kf)

    if C.op in s.outputs:
        Out = C
    else:
        Out = s.outputs[0].output(0)
        s[C].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], tvm.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], tvm.thread_axis("blockIdx.x"))

    tx = s[C].op.reduce_axis[0]
    thread_x = tvm.thread_axis("threadIdx.x")
    s[C].bind(tx, thread_x)
    s[CF].compute_at(s[C], tx)
    s[C].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))

def schedule_dense_large_batch(cfg, s, C):
    """Schedule float32/64 dense with large batch size"""
    A, B = C.op.input_tensors
    batch, in_dim = get_const_tuple(A.shape)
    out_dim, _ = get_const_tuple(B.shape)
    k = C.op.reduce_axis[0]

    # create tuning space
    try:
        block_cand = [64, 128]
        vthread_cand = [2**x for x in range(1, 7)]
        n_thread_cand = [2**x for x in range(3, 7)]
        cfg.define_split('tile_x', batch, num_outputs=4,
                         filter=lambda x: (x.size[1] in vthread_cand and
                                           x.size[2] in n_thread_cand and
                                           (x.size[1] * x.size[2] * x.size[3]) in block_cand))
        cfg.define_split('tile_y', out_dim, num_outputs=4,
                         filter=lambda x: (x.size[1] in vthread_cand and
                                           x.size[2] in n_thread_cand and
                                           (x.size[1] * x.size[2] * x.size[3]) in block_cand))
        cfg.define_split('tile_k', in_dim, num_outputs=3, filter=lambda x: x.size[0] > 2)
    except IndexError:
        # Index error happens when no entities left after filtering, which was designed
        # to prune tuning space for better search efficiency.
        logger.debug(
            'Tuning space was created without pruning due to unfit shapes')
        cfg.define_split('tile_x', batch, num_outputs=4)
        cfg.define_split('tile_y', out_dim, num_outputs=4)
        cfg.define_split('tile_k', in_dim, num_outputs=3)

    if cfg.is_fallback:
        if batch > 1:
            cfg['tile_x'] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg['tile_x'] = SplitEntity([1, 1, 1, 1])
        if out_dim > 1:
            cfg['tile_y'] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg['tile_y'] = SplitEntity([1, 1, 1, 1])
        if in_dim > 8:
            cfg['tile_k'] = SplitEntity([-1, 8, 1])
        else:
            cfg['tile_k'] = SplitEntity([-1, 1, 1])

    # Explicit memory access
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    # Deal with op fusion
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # Split and reorder computation
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, C.op.axis[0])
    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, C.op.axis[1])
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    # Binding
    s[C].bind(by, tvm.thread_axis("blockIdx.y"))
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tyz, tvm.thread_axis("vthread"))
    s[C].bind(txz, tvm.thread_axis("vthread"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    # Split reduction
    yo, xo = CC.op.axis
    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)

    # Schedule for A's shared memory load
    num_thread_x = cfg['tile_x'].size[2]
    ty, _ = s[AA].split(s[AA].op.axis[0], nparts=num_thread_x)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread_x * 4)
    tx, xi = s[AA].split(xi, nparts=num_thread_x)
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[AA].double_buffer()

    # Schedule for B' shared memory load
    num_thread_y = cfg['tile_y'].size[2]
    ty, _ = s[BB].split(s[BB].op.axis[0], nparts=num_thread_y)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread_y * 4)
    tx, xi = s[BB].split(xi, nparts=num_thread_y)
    s[BB].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[BB].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[BB].double_buffer()

