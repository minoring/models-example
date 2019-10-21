"""Public interface for flag definition.

See _example.py for detailed instructions on defining flags. 
TODO(minoring): There is no such file _example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from six.moves import shlex_quote

from absl import app as absl_app
from absl import flags

from official.utils.flags import _base
from official.utils.flags import _benchmark
from official.utils.flags import _conventions
from official.utils.flags import _device
from official.utils.flags import _distribution
from official.utils.flags import _misc
from official.utils.flags import _performance

