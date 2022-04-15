from invoke import Collection

from . import java
from . import conda
from . import pip

namespace = Collection(java, conda, pip)
