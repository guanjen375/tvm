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

# pylint: disable=redefined-builtin, wildcard-import
"""snps_ev specific declaration and schedules."""
from __future__ import absolute_import as _abs

from . import depthwise_conv2d, conv2d_transpose_nchw, deformable_conv2d, group_conv2d_nchw
from . import conv3d
from .conv2d import *
from .dense import *
from .nn import *
from .reduction import schedule_reduce
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .pooling import *
from .batch_matmul import schedule_batch_matmul
from .vision import *
from . import ssd
from .ssd import *
from .nms import *
from .sort import *
