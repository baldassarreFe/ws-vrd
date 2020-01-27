import pytest

import torch.testing


@pytest.fixture(params=torch.testing.get_all_device_types())
def device(request):
    return request.param
