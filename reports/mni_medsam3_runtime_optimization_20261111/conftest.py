"""Initialize PyTorch before the frozen retention tests patch sys.modules."""

# Checkpointing lazily imports Dynamo. A patch.dict(sys.modules) teardown would
# remove those Python modules while leaving their native operators registered.
import torch._dynamo  # noqa: F401
