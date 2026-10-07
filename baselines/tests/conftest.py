import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# The env code imports its own modules as top-level craftax_coop.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "craftax")))
