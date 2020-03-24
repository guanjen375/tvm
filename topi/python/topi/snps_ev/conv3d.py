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
"""Compute definition for conv3d with snps_ev backend"""
import tvm
from tvm import autotvm
from tvm.contrib import cudnn

from .. import nn, generic
from ..util import get_const_tuple, traverse_inline

from .conv3d_direct import schedule_direct_3d_snps_ev


@autotvm.register_topi_compute(nn.conv3d, ['snps_ev'], ['direct'])
def conv3d_snps_ev(cfg, data, kernel, strides, padding, dilation, layout='NCDHW', out_dtype='float32'):
    """Conv3D operator for snps_ev backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    kernel : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    target = tvm.target.current_target()

    if layout == 'NCDHW':
        return nn.conv3d_ncdhw(data, kernel, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv3d_ncdhw, ["snps_ev"],
                                ["direct"])
def schedule_conv3d_ncdhw_snps_ev(cfg, outs):
    """TOPI schedule callback of conv3d for snps_ev

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    target = tvm.target.current_target()

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv3d_ncdhw':
            schedule_direct_3d_snps_ev(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
