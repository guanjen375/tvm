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
# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d with auto fusion"""
import tvm
from tvm import autotvm
from ..util import traverse_inline
from .. import tag
from .. import generic, nn

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
autotvm.register_topi_compute(nn.depthwise_conv2d_nchw, ['snps_ev'], 'direct',
                              nn.depthwise_conv2d_nchw.fdefault)

@autotvm.register_topi_schedule(generic.schedule_depthwise_conv2d_nchw, ['snps_ev'], 'direct')
def schedule_depthwise_conv2d_nchw_snps_ev(cfg, outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'depthwise_conv2d_nchw':
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            cfg.define_split("tile_f", f, num_outputs=4)
            cfg.define_split("tile_y", y, num_outputs=4)
            cfg.define_split("tile_x", x, num_outputs=4)
            cfg.define_knob("auto_unroll_max_step", [0, 256, 1500])

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            # fallback support
            if cfg.is_fallback:
                ref_log = autotvm.tophub.load_reference_log(
                    target.target_name, target.model, 'depthwise_conv2d_nchw', 'direct')
                cfg.fallback_with_reference_log(ref_log)
                # TODO(lmzheng): A bug here, set unroll_explicit to False as workaround
                cfg['unroll_explicit'].val = 0
            ##### space definition end #####

            s[pad_data].compute_inline()
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, 'local')
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope('local')
                OL = conv

            # create cache stage
            AA = s.cache_read(pad_data, 'shared', [OL])
            WW = s.cache_read(kernel, 'shared', [OL])
            AL = s.cache_read(AA, 'local', [OL])
            WL = s.cache_read(WW, 'local', [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            kernel_scope, n = s[output].split(n, nparts=1)
            bf = s[output].fuse(n, bf)
            s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
            s[output].bind(by, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vy, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
            s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
            s[OL].compute_at(s[output], tx)

            # cooperative fetching
            s[AA].compute_at(s[output], bx)
            s[WW].compute_at(s[output], bx)
            s[AL].compute_at(s[output], tx)
            s[WL].compute_at(s[output], tx)

            for load in [AA, WW]:
                fused = s[load].fuse(*list(s[load].op.axis))
                fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
                fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
                fused, tz = s[load].split(fused, cfg["tile_f"].size[2])
                s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
                s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)
    return s

