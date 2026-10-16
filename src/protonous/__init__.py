"""protonous: a lightweight agent framework over OpenRouter.

Everything an agent is built from, importable from the top level:

    from protonous import Model, OpenrouterClient, OpenrouterConfig, Toolbox, get_settings

Lower-level pieces stay in their modules: `protonous.tools` (by_name, execute),
`protonous.normalizers` (the provider wire format), `protonous.cli`.
"""
from .client import OpenrouterClient, OpenrouterConfig, collect
from .context import (
    Message,
    MessageChunk,
    ReasoningDetail,
    Role,
    Step,
    ToolCall,
    ToolSpec,
    Usage,
    check_pairing,
)
from .errors import (
    ConfigurationError,
    LayerError,
    NotASessionError,
    ProtonousError,
    ProviderError,
    RunExistsError,
    SessionExistsError,
    StreamIncompleteError,
    ToolDefinitionError,
    ToolError,
    ToolPairingError,
    TraceFormatError,
)
from .layers import ContextWindow, Layer, SingleMessageLayer, SlidingWindowLayer, StaticLayer
from .model import Model, ReplyStream
from .settings import Settings, get_settings
from .tools import Tool, Toolbox, run_tools, tool
from .trace import (
    ErrorRecord,
    RunEnded,
    RunError,
    RunFailed,
    RunStarted,
    Session,
    SessionFollower,
    SessionTrace,
    StepRecorded,
    Trace,
    TraceEvent,
    Tracer,
    load_session,
    load_trace,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigurationError",
    "ContextWindow",
    "ErrorRecord",
    "Layer",
    "LayerError",
    "Message",
    "MessageChunk",
    "Model",
    "NotASessionError",
    "OpenrouterClient",
    "OpenrouterConfig",
    "ProtonousError",
    "ProviderError",
    "ReasoningDetail",
    "ReplyStream",
    "Role",
    "RunEnded",
    "RunError",
    "RunExistsError",
    "RunFailed",
    "RunStarted",
    "Session",
    "SessionExistsError",
    "SessionFollower",
    "SessionTrace",
    "Settings",
    "SingleMessageLayer",
    "SlidingWindowLayer",
    "StaticLayer",
    "Step",
    "StepRecorded",
    "StreamIncompleteError",
    "Tool",
    "ToolCall",
    "ToolDefinitionError",
    "ToolError",
    "ToolPairingError",
    "ToolSpec",
    "Toolbox",
    "Trace",
    "TraceEvent",
    "TraceFormatError",
    "Tracer",
    "Usage",
    "check_pairing",
    "collect",
    "get_settings",
    "load_session",
    "load_trace",
    "run_tools",
    "tool",
]
