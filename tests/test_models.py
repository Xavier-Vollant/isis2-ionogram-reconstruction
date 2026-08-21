import pytest

from isis_research.models import model_constructor


def test_model_constructor_rejects_unknown_names_without_loading_torch():
    with pytest.raises(ValueError, match="unsupported model"):
        model_constructor("does_not_exist")
