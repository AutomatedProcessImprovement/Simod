import traceback
from typing import Tuple

from hyperopt import STATUS_OK, STATUS_FAIL


class HyperoptPipeline:
    def step(self, status: str, fn, *args) -> Tuple[str, object]:
        """Function executes the provided function with arguments in hyperopt safe way."""
        if status == STATUS_OK:
            try:
                return STATUS_OK, fn(*args)
            except Exception as error:
                print(error)
                traceback.print_exc()
                return STATUS_FAIL, None
        else:
            return status, None
