from invoke import Collection

from . import conda
from . import java
from . import pip
from . import test

namespace = Collection(java, conda, pip, test)
