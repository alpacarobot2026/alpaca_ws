#!/usr/bin/env python3
"""
GuideBot interaction pipeline

Key TO KNOW for my dumbass:
- Using `awaiting_command` flag
- Core stuff:
    * STT muting while TTS is speaking (prevents the stupid pc from hearing itself).
    * Whisper stt.
    * LLM JSON parsing.
- Other:
    * Wake word → prompt → next utterance → LLM → prints room + coords.
"""

import sys
import numpy as np
import sounddevice as sd
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from visualization_msgs.msg import MarkerArray
try:
    from alpaca_interaction.face_speaker import FaceSpeaker
except Exception:
    try:
        from face_speaker import FaceSpeaker
    except Exception:
        FaceSpeaker = None
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from rclpy.node import Node
from rclpy.action import ActionClient
import rclpy

# Import nav2 action for path computation
try:
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from action_msgs.msg import GoalStatus
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False
    print("[WARN] nav2_msgs not found. Path validation will be disabled.")

import os
import json
import base64
import asyncio
import subprocess
import shutil
import argparse
from typing import List, Optional
import time
import threading
from collections import deque
from enum import Enum
from datetime import datetime
from pathlib import Path

import websockets
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import re
from difflib import SequenceMatcher
ALPACA_FACE_IMPORT_ERROR = None
try:
    from alpaca_interaction.alpaca_face_display import AlpacaFaceDisplay
except Exception as e_pkg:
    try:
        # Fallback when run as a script from inside package directory
        from alpaca_face_display import AlpacaFaceDisplay
    except Exception as e_local:
        AlpacaFaceDisplay = None
        ALPACA_FACE_IMPORT_ERROR = f"package import: {e_pkg}; local import: {e_local}"

# -----------------------------
# CONFIGURATION
# -----------------------------

# SAMPLE_RATE = 16000
# CHUNK_DURATION = 6   # seconds per audio chunk
# CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
# RMS_THRESHOLD = 0.003



### for third floor 
third_floor_location_info = {
    "3248": {
        'coordinates': [28.1526800038002, -49.186115393832],
        'yaw': -1.23377325,
        'tags': ['Faculty Office', 'Office', 'Room 3248']
    },

    "3220":{
        'coordinates': [59.088614078811126, -37.813810145858604],
        'yaw': 0.3370230786,
        'tags': ['Conference Room', 'Meeting Room', 'Room 3200']
    },

    # "3170":{
    #     'coordinates': [9.761987303931624, -24.157736595058083],
    #     'yaw': -1.34390352,
    #     'tags': ['Lab B', 'room 3170']
    # },
    "3269":{
        'coordinates': [14.654721710159107,-53.49407578712642],
        'yaw': -1.34390352,
        'tags': ['meeting room', 'privacy room', 'room 3269']
    },

    # "Desk": {
    #     'coordinates': [-1.7687367002236951, -7.226140391328384],
    #     'yaw': 0.0,
    #     'tags': ['Desk', 'Workstation']
    # },
    "Men's Room": {
        'coordinates': [43.219380106210764, -29.732895103461],
        'yaw': 0.3370230786,
        'tags': ['Men\'s Room', 'Bathroom', 'Restroom']
    },
    "3120": {
        'coordinates': [46.89909118768621, -39.194109683014],
        'yaw': 0.3370230786,
        'tags': ['Room 3120','Office Supplies', 'Printing room']
    },
    "Women's Room": {
        'coordinates': [40.58667903049457, -22.253996348498],
        'yaw': 0.3370230786,
        'tags': ['Women\'s Room', 'Bathroom', 'Restroom']
    },
    "Main Elevator": {
        'coordinates': [40.97095131977092,-14.17125377728],
        'yaw': -1.23377325,
        'tags': ['Main Elevator', 'Elevator']
    },
    "3225": {
        'coordinates': [43.98982382247678, -43.573055956524],
        'yaw': 1.9078194054,
        'tags': ['Room 3225','Conference Room', 'Meeting Room']
    },
    "3224": {
        'coordinates': [43.98982382247678, -43.573055956524],
        'yaw': -1.23377325,
        'tags': ['Room 3224', 'HR Office']
    },
    "water fountain": {
        'coordinates': [39.1662802030454,-19.10017335930586],
        'yaw': 1,
        'tags': ['water fountain', 'drinking fountain', 'water', 'thirsty']
    },
    "Kitchen": {
        'coordinates': [40.18143459800027, -44.826641963787],
        'yaw':  1.9078194054,
        'tags': ['Room 3229', 'Kitchen', 'Break Room', 'Refrigerator', 'Snacks', 'Coffee', 'Microwave']
    },
}


### for first floor###
first_floor_location_info = {

    "Cafe": {
        'coordinates': [13.47198766185206, -25.948669821649933],
        'yaw': 0.0,
        'tags': ['Cafe', 'Coffee', 'Snack', 'Breakfast', 'Lunch', 'Pastries', 'Tea']
    },

    "Main Entrance": {
        'coordinates': [12.923670773030349 ,-32.306569012124804],
        'yaw': -0.890118,
        'tags': ['Main Exit', 'Front Door']
    },

    "Lab A": {
        'coordinates': [-2.770766912662747, -19.75275935172515],
        'yaw': -1.57,
        'tags': ['Lab A', 'Robotics Lab', 'Room 1020']
    },
    "Room 1060": {
        'coordinates': [-2.0237115883835277 ,7.860415568319996],
        'yaw': 3.08923,
        'tags': ['Classroom 1060']
    },
    "Room 1050":{
        'coordinates': [-2.4624465526916834,-1.202643564074761],
        'yaw': 3.08923,
        'tags': ['Classroom 1050']
    },
    "Back Exit":{
        'coordinates': [-2.299803344235273, 18.673274381239953],
        'yaw': 1.39626,
        'tags': ['Back Door', 'Back Entrance']
    },

    "Stage":{
        'coordinates': [9.81368716685953, -6.64533123583268],
        'yaw': -1.39626,
        'tags': ['Couch', 'TV area', 'Stage area', 'podium']
    }


}

task_info = {
    "location_info": first_floor_location_info
}


def _build_location_options(task_info_obj: dict) -> tuple[list[str], list[str]]:
    valid = list((task_info_obj or {}).get("location_info", {}).keys())
    special = ["other", "multiple", "na"]
    return valid, special


def _build_location_enum(valid_locations: list[str], special_locations: list[str]) -> type[Enum]:
    members = {}
    used_names = set()

    def enum_name_for(label: str, index: int) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper() or f"LOC_{index}"
        if name[0].isdigit():
            name = f"LOC_{name}"
        base = name
        suffix = 2
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)
        return name

    for i, location in enumerate(valid_locations):
        members[enum_name_for(location, i)] = location

    for i, location in enumerate(special_locations):
        members[enum_name_for(location, i + len(valid_locations))] = location

    return Enum("LocationOption", members, type=str)


VALID_LOCATIONS, SPECIAL_LOCATION_TOKENS = _build_location_options(task_info)
LocationOption = _build_location_enum(VALID_LOCATIONS, SPECIAL_LOCATION_TOKENS)
ALL_LOCATION_OPTIONS = VALID_LOCATIONS + SPECIAL_LOCATION_TOKENS
DialogState = Literal[
    "idle",
    "awaiting_navigation_confirmation",
    "awaiting_location_clarification",
    "navigating",
]


class RobotState(Enum):
    """High-level operating mode; gates face_speaker vs nav2 control of cmd_vel."""
    IDLE = "idle"          # Dormant; awaiting wake word; face_speaker active
    AWAKE = "awake"        # Wake word heard; listening/conversing; face_speaker active
    GUIDING = "guiding"    # Nav2 navigating to destination; face_speaker disabled
    ARRIVED = "arrived"    # Goal reached; brief state before returning to IDLE

# task_info = {
#   "app_name": "MyVoiceAgent",
#   "policies": {
#     "style": "concise, direct",
#     "safety": "refuse disallowed requests"
#   },
#   "domain_knowledge": {
#     "supported_tasks": ["qa", "summarize", "plan", "troubleshoot"]
#   }
# }

system_prompt = """
# ROLE
You are a voice-enabled guide robot named "Alpaca" in a university robotics building. Your purpose is to physically guide users to destinations within the building.

The user voice input translated to text will be provided in the 'user_text' field ((may contain automatic speech recognition errors)).

You will be provided with a JSON object called 'task_info' that contains location identifiers for valid locations you can guide them to in the building.
Each identifier is a string which corresponds to a dictionary with:
- 'coordinates': the location's position
- 'tags': keywords associated with the location name

# SEMANTIC REASONING
IMPORTANT: You must reason about what the user WANTS TO DO, not just what location they mention.

Examples of intent → location mapping:
- "I need to take a meeting" → Conference Room (3120 or 3225)
- "I'm hungry" → Kitchen or Cafe
- "I need to make a call" → Conference Room (for privacy)
- "Where can I get coffee?" → Kitchen or Cafe
- "I need to use the restroom" → Men's Room or Women's Room
- "I have an interview" → Conference Room
- "I need a quiet place to work" → Conference Room or Office

Match the user's INTENT/ACTIVITY to locations by checking the 'tags' field and location identifiers in task_info.
If the user's request implies an activity that matches a location, SUGGEST that location proactively.

# CAPABILITIES
- Guide users to known locations (listed in TASK_INFO_JSON)
- Infer user intent and suggest appropriate locations
- Answer questions about the building
- Use web search for up-to-date information when needed

Note: your downstream task will be to take the user to the destination they want after you have identified it. Whenever you ask for confirmation, make clear you will physically guide them there — do not just give verbal directions.

# ROUTING RULES
Set all output fields explicitly on every turn. Use these rules based on what the user said:

**User explicitly asks to be taken/guided/led/navigated to a single valid location:**
'location' = that location identifier, 'next_action' = 'navigate', 'dialog_state' = 'navigating', 'pending_location' = 'na'. Confirm you are guiding them now.

**User asks about a single valid location but has NOT explicitly requested guidance:**
'location' = that location identifier, 'next_action' = 'speak', 'dialog_state' = 'awaiting_navigation_confirmation', 'pending_location' = that location identifier. Briefly describe or confirm the location and ask if they'd like to be guided there.

**Multiple valid locations match the request:**
'location' = 'multiple', 'next_action' = 'speak', 'dialog_state' = 'awaiting_location_clarification', 'pending_location' = 'multiple'. Name the options and ask which they prefer.

**Location not in the valid list:**
'location' = 'other', 'next_action' = 'speak', 'dialog_state' = 'idle', 'pending_location' = 'na'. Apologize that you cannot guide them there. If the location is inside the building, give a brief description of where to find it.

**Intent is 'question' (a general question, not about finding a destination):**
'location' = 'na', 'next_action' = 'speak', 'dialog_state' = 'idle', 'pending_location' = 'na'. Answer helpfully and concisely. Use web search if needed.

**Intent is 'other' (not a question and not about a destination):**
'location' = 'na', 'next_action' = 'speak', 'dialog_state' = 'idle', 'pending_location' = 'na'. Politely say you can only help with navigating the building.

# DIALOG STATE TRANSITIONS
DIALOG_STATE_JSON is the authoritative source of the current conversation state. Read it before generating a response; update 'dialog_state' and 'pending_location' in your output on every turn.

**If dialog_state is 'idle':** Apply routing rules above.

**If dialog_state is 'awaiting_navigation_confirmation':**
The user is responding to your earlier question about guiding them to DIALOG_STATE_JSON.pending_location.
1. Accept (yes / sure / okay / any affirmative): 'location' = pending_location, 'next_action' = 'navigate', 'dialog_state' = 'navigating', 'pending_location' = 'na'.
2. Decline (no / never mind / etc.): 'location' = 'na', 'next_action' = 'speak', 'dialog_state' = 'idle', 'pending_location' = 'na'. Briefly acknowledge and offer to help with something else.
3. New destination requested: treat as a fresh request; apply routing rules above.
4. Unclear: ask one short clarifying question; keep 'dialog_state' as 'awaiting_navigation_confirmation' and 'pending_location' unchanged.

**If dialog_state is 'awaiting_location_clarification':**
The user is choosing among the options you previously listed.
1. Clear single selection: apply routing rules for that location.
2. Still ambiguous: ask one more short clarifying question; keep 'dialog_state' as 'awaiting_location_clarification'.
3. Opts out: 'location' = 'na', 'next_action' = 'speak', 'dialog_state' = 'idle', 'pending_location' = 'na'.

**If dialog_state is 'navigating':**
The robot is actively guiding the user. Do NOT change 'dialog_state' away from 'navigating' on a speak-only response.
1. Cancel/stop request: acknowledge ("Stopping now."), 'next_action' = 'speak', keep 'dialog_state' = 'navigating' (the system handles the actual stop).
2. Question about the building, labs, researchers, etc.: answer in 1–2 sentences; keep 'dialog_state' = 'navigating', 'pending_location' = 'na'.
3. New destination request: confirm the new destination, 'next_action' = 'speak', 'dialog_state' = 'awaiting_navigation_confirmation', 'pending_location' = new location.

# INTENT CLASSIFICATION
'intent' = 'find destination' — user is asking about or requesting a specific destination.
'intent' = 'question' — user asks a general question not related to finding a destination.
'intent' = 'other' — anything else.

If you need up-to-date facts, use web search.
This prompt may be called multiple times in a conversation. Use the message history to understand context.

You are to produce output strictly in the provided schema. All responses should be in English only, even if the user's input is in another language. Keep spoken responses under 2 short sentences (about 300 characters) unless the user asks for more detail.

# BEING TALKED TO
Determine whether the transcript is addressed to you or is background conversation. Set 'being_talked_to' accordingly.
Signals you ARE being addressed:
- Direct address: "Hey Alpaca", "Excuse me, can you...", name + request
- Content is a request or question that makes sense directed at a guide robot
- Follows a recent interaction (within 30 seconds of your last response)

When 'being_talked_to' is false: set 'answer' to "", 'location' to 'na', 'next_action' to 'speak', 'pending_location' to 'na', and leave 'dialog_state' unchanged (preserve its current value from DIALOG_STATE_JSON).


# ERROR HANDLING
- If speech recognition seems garbled: Ask user to repeat more slowly
- User will only be talking in English, ignore text in other languages
- If location is ambiguous: List options and ask for clarification
- If unsure about intent: Ask a clarifying question rather than guessing


# EXAMPLES
Example 1 - Clear navigation request:
User: "Where can I find the bathroom?"
→ intent: "find destination"
→ location: "multiple" (Men's Room and Women's Room both match)
→ answer: "There's a men's room and a women's room nearby. Which would you prefer?"
→ next_action: "speak"
→ dialog_state: "awaiting_location_clarification"
→ pending_location: "multiple"
→ being_talked_to: true
Example 2 - Confirmation to navigate:
User: "Yes, take me to the men's room"
→ intent: "find destination"
→ location: "Men's Room"
→ answer: "Great! I'll guide you to the men's room now. Please follow me."
→ next_action: "navigate"
→ dialog_state: "navigating"
→ pending_location: "na"
→ being_talked_to: true
Example 3 - Background conversation (not addressed):
User: "...and then I told him the elevator was broken and she asked me to take stairs but i was too tired..."
→ intent: "other"
→ location: "na"
→ answer: ""
→ next_action: "speak"
→ dialog_state: "idle"
→ pending_location: "na"
→ being_talked_to: false
Example 4 - Non-navigation question requiring web search:
User: "What time does Cafe close?"
→ intent: "question"
→ location: "na"
→ answer: "Cafe is open today until 6:00 PM."
→ next_action: "speak"
→ dialog_state: "idle"
→ pending_location: "na"
→ being_talked_to: true
→ used_web_search: true
Example 5 - Single location found, confirmation pending:
User: "Where’s the kitchen?"
→ intent: "find destination"
→ location: "Kitchen"
→ answer: "The kitchen’s just down the hall. Want me to take you there?"
→ next_action: "speak"
→ dialog_state: "awaiting_navigation_confirmation"
→ pending_location: "Kitchen"
→ being_talked_to: true
Example 6 - General question (not a destination):
User: "Who runs Lab A?"
→ intent: "question"
→ location: "na"
→ answer: "Lab A is on the first floor."
→ next_action: "speak"
→ dialog_state: "idle"
→ pending_location: "na"
→ being_talked_to: true

NOTE: If web search is required, you MUST use web search before generating the answer and return the final answer directly and not use any placeholders. Never output "I’ll look it up / let me check / please hold on".
If web search is needed, perform it internally and return the final answer in the same response. Never include links or URLs in your response.
"""


MAX_TTS_CHARS = 300

# -----------------------------
# 1) Define your required output JSON schema
# -----------------------------
class AssistantOutput(BaseModel):
    model_config = {"use_enum_values": True}
    # Customize these fields to match your "prespecified json format"
    intent: Literal["question", "find destination", "other"] = Field(description="What the assistant decided the user wants. Should fit into three categories: question if the user is asking a question NOT related to finding a destination, find destination if the user is asking about a specific destination, or other otherwise.")
    location: LocationOption = Field(description=f"The location the user is asking about. Must be one of: {', '.join(ALL_LOCATION_OPTIONS)}")
    answer: str = Field(description="The main response to the user.")
    next_action: Literal["navigate", "speak"] = Field(description="The next action you will take. Navigate if you are to guide the user to the location. Speak if you are to respond to the user.")
    dialog_state: DialogState = Field(description="The dialog state to store for the next user turn.")
    pending_location: LocationOption = Field(description=f"The pending destination for the next turn. Use a valid location, 'multiple', or 'na'. Must be one of: {', '.join(ALL_LOCATION_OPTIONS)}")
    used_web_search: bool = Field(description="Whether web_search was used.")
    being_talked_to: bool = Field(description="Whether the user is currently talking to the guide robot.")


# -----------------------------
# 2) Helpers
# -----------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def b64encode_audio_pcm16(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")

def pcm16_resample_linear(x_int16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample 1D int16 PCM from src_sr to dst_sr using linear interpolation."""
    if src_sr == dst_sr:
        return x_int16

    x = x_int16.astype(np.float32)
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_sr / src_sr)))
    if n_src < 2 or n_dst < 2:
        return x_int16

    src_idx = np.linspace(0, n_src - 1, num=n_dst, dtype=np.float32)
    x_dst = np.interp(src_idx, np.arange(n_src, dtype=np.float32), x)
    x_dst = np.clip(np.rint(x_dst), -32768, 32767).astype(np.int16)
    return x_dst


def sanitize_for_tts(s: str) -> str:
    """Remove text that should never be read aloud: URLs, bare domains, e-mail
    addresses, markdown links/citations and web-search citation markers.
    The web_search tool makes the model append sources to its answer."""
    s = s or ""
    # markdown links [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)\s]+\)", r"\1", s)
    # citation markers like [1], [2, 3], ([source]) and "(source: ...)" parentheticals
    s = re.sub(r"\[(?:\d+(?:\s*,\s*\d+)*|source|citation)\]", "", s, flags=re.I)
    s = re.sub(r"\((?:source|sources|citation|ref)[^)]*\)", "", s, flags=re.I)
    # full URLs (with or without scheme) and e-mail addresses
    s = re.sub(r"(?:https?://|www\.)\S+", "", s, flags=re.I)
    s = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", "", s)
    # bare domains such as example.edu or example.com/path
    s = re.sub(r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|us|uk|de|ca)\b(?:/\S*)?", "", s, flags=re.I)
    # leftover empty brackets/parens and dangling punctuation
    s = re.sub(r"\(\s*\)|\[\s*\]", "", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,.;:])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    # Normalize whitespace so character budgeting is more predictable for TTS.
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s

    budget = max_chars - 1  # reserve room for ellipsis if needed

    # 1) Prefer full-sentence packing.
    sentence_chunks = re.findall(r"[^.!?]+[.!?]?", s)
    packed = ""
    for chunk in sentence_chunks:
        c = chunk.strip()
        if not c:
            continue
        candidate = (packed + " " + c).strip() if packed else c
        if len(candidate) <= max_chars:
            packed = candidate
        else:
            break
    if packed and len(packed) >= int(0.45 * max_chars):
        return packed.rstrip()

    # 2) Fallback to the best punctuation boundary within budget.
    s2 = s[:budget].rstrip()
    punct_positions = [m.end() for m in re.finditer(r"[.!?;:]\s", s2)]
    if punct_positions:
        cut = punct_positions[-1]
        if cut >= int(0.45 * budget):
            out = s2[:cut].rstrip()
            if out and out[-1] not in ".!?":
                out += "."
            return out

    # 3) Fallback to word boundary + ellipsis, avoiding dangling connectors.
    for sep in [" and ", " but ", " so ", ",", ";", ":", "—", "-"]:
        idx = s2.rfind(sep)
        if idx >= int(0.52 * budget):
            out = s2[:idx].rstrip(" ,;:-—")
            return out + "..."

    sp = s2.rfind(" ")
    if sp >= int(0.50 * budget):
        return s2[:sp].rstrip() + "..."

    return s2.rstrip() + "..."

# -----------------------------
# 3) ROS2 Node for Navigation Integration
# -----------------------------
class AlpacaInteractionNode(Node):
    """ROS2 node for path validation and navigation goal publishing."""
    
    def __init__(self):
        super().__init__('alpaca_interaction_node')
        
        # Publisher for navigation goal
        self.coord_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Action client for path computation (path validation before navigation)
        if NAV2_AVAILABLE:
            self._compute_path_client = ActionClient(
                self, ComputePathToPose, '/compute_path_to_pose'
            )
            self._navigate_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        else:
            self._compute_path_client = None
            self._navigate_client = None
        self._cancel_plan_client = self.create_client(Trigger, '/cancel_navigation')
        self._stop_mppi_client = self.create_client(Trigger, '/stop_navigation')
        self._current_goal_handle =  None
        self._current_location_name: Optional[str] = None
        self._goal_reached = False
        self._goal_reached_callback = None
        self._last_nav_distance_remaining: float = float('inf')
        self._motion_hint_callback = None
        self._rollout_preview_callback = None

        # Reuse existing navigation topics to infer anticipatory turn direction.
        self._cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self._rollouts_sub = self.create_subscription(
            MarkerArray, '/mpc_local_planner_mppi/vis/rollouts', self._rollouts_cb, 10
        )
        # Compatibility fallback: some setups visualize/publish on /mppi_rollouts.
        self._rollouts_sub_legacy = self.create_subscription(
            MarkerArray, '/mppi_rollouts', self._rollouts_cb, 10
        )
        
        self.get_logger().info("AlpacaInteractionNode initialized")

    def set_goal_reached_callback(self,callback):
        self._goal_reached_callback = callback
    def set_motion_hint_callback(self, callback):
        self._motion_hint_callback = callback
    def set_rollout_preview_callback(self, callback):
        self._rollout_preview_callback = callback

    def _emit_motion_hint(self, source: str, value: float):
        if self._motion_hint_callback is None:
            return
        try:
            self._motion_hint_callback(source, value)
        except Exception:
            pass

    def _emit_rollout_preview(self, points):
        if self._rollout_preview_callback is None:
            return
        try:
            self._rollout_preview_callback(points)
        except Exception:
            pass

    def _cmd_vel_cb(self, msg: Twist):
        omega = float(msg.angular.z)
        omega_ref = 0.8
        gaze = max(-1.0, min(1.0, omega / omega_ref))
        self._emit_motion_hint("cmd", gaze)

    def _rollouts_cb(self, msg: MarkerArray):
        # Estimate signed turn direction from near-horizon trajectory curvature.
        vals = []
        previews = []
        for m in msg.markers:
            pts = m.points
            if pts is None or len(pts) < 3:
                continue
            i1 = min(2, len(pts) - 1)
            i2 = min(5, len(pts) - 1)
            if i2 <= i1:
                continue
            p0 = pts[0]
            p1 = pts[i1]
            p2 = pts[i2]
            v1x = float(p1.x - p0.x)
            v1y = float(p1.y - p0.y)
            v2x = float(p2.x - p1.x)
            v2y = float(p2.y - p1.y)
            n1 = float(np.hypot(v1x, v1y))
            n2 = float(np.hypot(v2x, v2y))
            if n1 < 1e-5 or n2 < 1e-5:
                continue
            cross = v1x * v2y - v1y * v2x
            signed_curve = cross / (n1 * n2 + 1e-6)
            # Amplify subtle curvature so gentle bends still move pupils.
            curve_gain = 2.6
            val = max(-1.0, min(1.0, float(curve_gain * signed_curve)))
            vals.append(val)
            previews.append(pts)

        if not vals:
            return

        # Use the strongest-turning rollout instead of median, which can cancel out.
        arr = np.asarray(vals, dtype=np.float32)
        best_idx = int(np.argmax(np.abs(arr)))
        gaze = float(arr[best_idx])
        # Suppress near-straight rollouts; strongly emphasize meaningful turns.
        straight_deadband = 0.18
        if abs(gaze) < straight_deadband:
            gaze = 0.0
        else:
            # Re-expand remaining range for clearer directional cue.
            mag = (abs(gaze) - straight_deadband) / (1.0 - straight_deadband)
            mag = max(0.0, min(1.0, mag))
            gaze = np.sign(gaze) * (0.35 + 0.65 * mag)
        self._emit_motion_hint("rollout", gaze)

        # Normalize selected rollout into [0,1]x[0,1] for UI preview.
        best_preview = previews[best_idx] if 0 <= best_idx < len(previews) else None
        if best_preview is not None and len(best_preview) >= 2:
            xs = np.asarray([float(p.x) for p in best_preview], dtype=np.float32)
            ys = np.asarray([float(p.y) for p in best_preview], dtype=np.float32)
            x0 = xs[0]
            y0 = ys[0]
            dx = xs - x0
            dy = ys - y0
            max_abs = float(max(np.max(np.abs(dx)), np.max(np.abs(dy)), 1e-3))
            xn = 0.5 + 0.45 * (dx / max_abs)
            yn = 0.1 + 0.8 * ((dy - np.min(dy)) / (np.max(dy) - np.min(dy) + 1e-6))
            preview = [(float(max(0.0, min(1.0, a))), float(max(0.0, min(1.0, b)))) for a, b in zip(xn, yn)]
            self._emit_rollout_preview(preview)

    def publish_goal(self, x: float, y: float, location_name: str, yaw: float = 0.0):
        """Publish a navigation goal to /goal_pose topic."""
        import math
        msg = PoseStamped()
        msg.header = Header()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.coord_pub.publish(msg)
        self.get_logger().info(f"Published goal: ({x}, {y})")
        # Store the location name for announcement when goal is reached
        self._current_location_name = location_name
        self._goal_reached = False
        
        # Also send goal via NavigateToPose action to track completion status
        if NAV2_AVAILABLE and self._navigate_client is not None:
            print(f"[NAV] Sending navigation goal to {location_name} via NavigateToPose action...")
            self._send_navigation_goal(msg)
    
    def _send_navigation_goal(self, goal_pose: PoseStamped):
        """Send navigation goal via NavigateToPose action to track completion."""
        if not self._navigate_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("NavigateToPose action server not available")
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.get_logger().info("Sending navigation goal to track completion...")
        send_goal_future = self._navigate_client.send_goal_async(
            goal_msg,
            feedback_callback=self._navigation_feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)
    
    def _goal_response_callback(self, future):
        """Handle the response when a navigation goal is accepted/rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                "NavigateToPose goal rejected by action server — "
                "goal completion will not be tracked via nav2 action. "
                "Ensure nav2 is not already executing a conflicting goal."
            )
            return

        self.get_logger().info("Navigation goal accepted")
        self._current_goal_handle = goal_handle
        self._last_nav_distance_remaining = float('inf')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)
    
    def _navigation_feedback_callback(self, feedback_msg):
        self._last_nav_distance_remaining = feedback_msg.feedback.distance_remaining
    
    def _goal_result_callback(self, future):
        """Handle the final result when navigation completes."""
        result = future.result()
        status = result.status
        dist = self._last_nav_distance_remaining
        print(f"[NAV] Navigation completed with status: {status}, last_distance_remaining={dist:.2f}m")

        goal_reached = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            goal_reached = True
           
        elif status == GoalStatus.STATUS_ABORTED:
            # MPC custom stack may reach goal before nav2 resolves its BT,
            # causing nav2 to ABORT rather than SUCCEED.  If distance
            # feedback shows we were within 1.0 m when the abort fired,
            # treat it as arrival.
            if dist < 1.0:
                self.get_logger().info(
                    f"Nav2 ABORTED but distance was {dist:.2f}m — treating as goal reached"
                )
                goal_reached = True
            else:
                self.get_logger().warn(
                    f"Navigation aborted at distance {dist:.2f}m (genuine failure)"
                )
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Navigation goal was canceled")
        else:
            self.get_logger().info(f"Navigation ended with status: {status}")

        if goal_reached:
            self.get_logger().info(f"Navigation goal reached: {self._current_location_name}")
            self._goal_reached = True
            future = self._stop_mppi_client.call_async(Trigger.Request())
            future.add_done_callback(lambda fut: self._log_trigger_result(fut, "/stop_navigation"))
            if self._goal_reached_callback is not None:
                self._goal_reached_callback(self._current_location_name)

        self._current_goal_handle = None
    
    def is_goal_reached(self) -> bool:
        """Check if the current navigation goal has been reached."""
        return self._goal_reached
    
    def get_current_location_name(self) -> Optional[str]:
        """Get the name of the current navigation target."""
        return self._current_location_name
    
    def cancel_navigation(self):
        """Cancel the current navigation goal if one is active."""
        # Stop local controller first so motion halts immediately.
        if self._stop_mppi_client.wait_for_service(timeout_sec=0.5):
            future = self._stop_mppi_client.call_async(Trigger.Request())
            future.add_done_callback(lambda fut: self._log_trigger_result(fut, "/stop_navigation"))
        else:
            self.get_logger().warn("/stop_navigation service not available")

        # Clear segmented plan so no stale /goal is republished.
        if self._cancel_plan_client.wait_for_service(timeout_sec=0.5):
            future = self._cancel_plan_client.call_async(Trigger.Request())
            future.add_done_callback(lambda fut: self._log_trigger_result(fut, "/cancel_navigation"))
        else:
            self.get_logger().warn("/cancel_navigation service not available")

        # Cancel nav2 action tracking handle.
        if self._current_goal_handle is not None:
            self.get_logger().info("Canceling navigation goal...")
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._goal_reached = False
        self._last_nav_distance_remaining = float('inf')

    def _log_trigger_result(self, future, service_name: str):
        try:
            res = future.result()
            if res is not None and res.success:
                self.get_logger().info(f"{service_name}: {res.message}")
            else:
                msg = res.message if res is not None else "empty response"
                self.get_logger().warn(f"{service_name} failed: {msg}")
        except Exception as e:
            self.get_logger().warn(f"{service_name} call exception: {e}")

    
    def compute_path_to_pose(self, x: float, y: float, yaw: float = 0.0, timeout_sec: float = 5.0) -> bool:
        """
        Use ComputePathToPose action to check if the goal is reachable.
        Returns True if a valid path is found, False otherwise.
        """
        import math
        if not NAV2_AVAILABLE or self._compute_path_client is None:
            self.get_logger().warn("Path validation unavailable (nav2_msgs not installed)")
            return True  # Assume reachable if we can't check

        # Wait for action server
        if not self._compute_path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("ComputePathToPose action server not available")
            return True  # Assume reachable if server unavailable

        # Create goal message
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = x
        goal_msg.goal.pose.position.y = y
        goal_msg.goal.pose.position.z = 0.0
        goal_msg.goal.pose.orientation.x = 0.0
        goal_msg.goal.pose.orientation.y = 0.0
        goal_msg.goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.goal.pose.orientation.w = math.cos(yaw / 2.0)
        goal_msg.planner_id = "GridBased"  # Match nav2_params planner
        goal_msg.use_start = False  # Use robot's current pose as start
        
        # Send goal and wait for result
        future = self._compute_path_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose goal rejected")
            return False
        
        # Get result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        
        result = result_future.result()
        if result is None:
            self.get_logger().warn("ComputePathToPose result timeout")
            return False
        
        # Check error code: 0 = success
        error_code = result.result.error_code
        if error_code == 0:
            path_len = len(result.result.path.poses)
            self.get_logger().info(f"Path found with {path_len} poses")
            return True
        else:
            self.get_logger().warn(f"No valid path found (error_code={error_code})")
            return False


# -----------------------------
# 4) Main "transcribe turn -> call GPT -> emit JSON" pipeline
# -----------------------------
class VoiceChatbot:
    def __init__(
        self,
        system_prompt_path: str,
        task_info_path: str,
        *,
        transcription_model: str = "gpt-4o-transcribe",
        response_model: str = "gpt-4o",
        sample_rate: int = 24000,
        frame_ms: int = 20,
        text_mode: bool = False,
        use_ros: bool = True,
        skip_path_check: bool = False,
        audio_backend: str = "auto",
        pulse_source: Optional[str] = None,
        pulse_sink: Optional[str] = None,
        sd_input_device: Optional[str] = None,
        sd_output_device: Optional[str] = None,
        latest_only: bool = False,
        use_wake_word: bool = False,
        wake_timeout: float = 60.0,
        wake_phrase: str = "hey alpaca",
        use_face_display: bool = False,
        face_fullscreen: bool = True,
        dummy_motion: bool = False,
        transcript_dir: str = "data",
        use_face_speaker: bool = False,
        vad_threshold: float = 0.65,
        noise_filter_hz: float = 0.0,
    ):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.task_info = task_info
        self._static_system_msg = self._build_static_system_msg()
        transcript_root = Path(transcript_dir).expanduser()
        run_stamp = datetime.now().strftime("transcript_%Y%m%d_%H%M%S")
        self.data_dir = transcript_root / run_stamp
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.data_dir / "conversation_transcript.txt"
        self._transcript_lock = threading.Lock()
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(f"# Conversation transcript started {datetime.now().isoformat()}\n")
        print(f"[LOG] Transcript file: {self.transcript_path}")

        self.transcription_model = transcription_model
        self.response_model = response_model

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

        # Conversation memory (optional; keep short in real apps)
        self.history: List[dict] = []

        # Streaming queues
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._utterance_q: asyncio.Queue[str] = asyncio.Queue()

        self.audio_backend = audio_backend
        self.pulse_source = pulse_source
        self.pulse_sink = pulse_sink
        self.sd_input_device = sd_input_device
        self.sd_output_device = sd_output_device
        self.latest_only = latest_only

        # Wake word detection (text-based)
        self.use_wake_word = use_wake_word
        self.wake_phrase = wake_phrase.lower()
        self.wake_timeout = wake_timeout
        self._wake_active = not use_wake_word  # If disabled, always active
        self._last_wake_time = 0.0
        self._last_wake_trigger_time = 0.0
        self._wake_debounce_sec = 0.45
        self._wake_threshold = 76.0
        self._wake_threshold_start = 68.0
        self._wake_start_window_tokens = 6
        self._wake_variants = self._build_wake_variants(self.wake_phrase)
        if self.use_wake_word:
            print(f"[WAKEWORD] Text-based wake word enabled. Say '{wake_phrase}' to activate.")
            # Keep wake window alive after each bot response; timeout counts from last model reply.
            self._last_wake_time = time.time()

        # Track ordering between commit and transcript completion
        self._latest_committed_item_id: Optional[str] = None
        self._seen_transcript_item_ids: set[str] = set()
        self._recent_transcripts = deque()
        self._transcript_dedupe_window_sec = 2.5

        # Text mode flag for debugging
        self.text_mode = text_mode

        self._dbg_last_audio_print = 0.0
        self._dbg_audio_chunks = 0

        self._mic_sr = None
        self._target_sr = 24000
        
        # ROS2 integration (optional)
        self.use_ros = use_ros
        self.ros_node: Optional[AlpacaInteractionNode] = None
        self._face_speaker_node: Optional[object] = None  # FaceSpeaker node
        if self.use_ros:
            try:
                if not rclpy.ok():
                    rclpy.init()
                self.ros_node = AlpacaInteractionNode()
                self.ros_node.set_goal_reached_callback(self._on_goal_reached)
                from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
                self._ros_executor = MultiThreadedExecutor()
                self._ros_executor.add_node(self.ros_node)

                self._ros_spin_thread = threading.Thread(
                    target=self._ros_executor.spin,
                    daemon=True
                )
                self._ros_spin_thread.start()
                print("[ROS] Spinning executor in background thread")

                # FaceSpeaker runs in its own SingleThreadedExecutor — matches original standalone behavior.
                if use_face_speaker and FaceSpeaker is not None:
                    self._face_speaker_node = FaceSpeaker()
                    # Start disabled; _transition_to / init block below will enable when appropriate.
                    self._face_speaker_node.set_enabled(False)
                    self._fs_executor = SingleThreadedExecutor()
                    self._fs_executor.add_node(self._face_speaker_node)
                    self._fs_spin_thread = threading.Thread(
                        target=self._fs_executor.spin,
                        daemon=True
                    )
                    self._fs_spin_thread.start()
                    print("[FACE_SPEAKER] FaceSpeaker node spinning on dedicated SingleThreadedExecutor")

                print("[INFO] ROS2 node initialized successfully")
            except Exception as e:
                print(f"[WARN] Failed to initialize ROS2: {e}")
                print("[WARN] Continuing without ROS2 navigation capabilities")
                self.ros_node = None
                self._face_speaker_node = None
                self.use_ros = False
        else:
            print("[INFO] ROS2 disabled (--no-ros flag)")
        
        self.vad_threshold = max(0.0, min(1.0, vad_threshold))
        self.noise_filter_hz = noise_filter_hz
        self._hpf_sos = None   # cached scipy high-pass filter coefficients
        self._hpf_sos_sr = 0   # sample rate the sos was built for

        # ---- State machine ----
        # robot_state is the authoritative FSM node.
        # self.state ("speaking" | "guiding") is kept in sync for backward compat.
        self.robot_state: RobotState = RobotState.IDLE
        self.state: str = "speaking"
        self._use_face_speaker = use_face_speaker

        self._current_navigation_location: Optional[str] = None
        self.dialog_state: DialogState = "idle"
        self.pending_location: str = "na"
        
        
        # Path validation option
        self.skip_path_check = skip_path_check
        
        # TTS muting flag - prevents STT from hearing TTS output
        self._tts_playing = False
        # Serialize TTS so async callbacks (for example goal-reached) cannot overlap playback.
        self._tts_lock = threading.Lock()
        self.dummy_motion = dummy_motion

        # Incremented on every cancel/arrival so _utterance_worker can detect stale LLM responses.
        self._nav_cancel_epoch = 0

        # Optional on-screen face display
        self.use_face_display = use_face_display and AlpacaFaceDisplay is not None
        self.face_display = None
        if use_face_display and AlpacaFaceDisplay is None:
            print("[WARN] Face display module import failed; continuing without face UI.")
            if ALPACA_FACE_IMPORT_ERROR:
                print(f"[WARN] Face import error details: {ALPACA_FACE_IMPORT_ERROR}")
        if self.use_face_display:
            try:
                self.face_display = AlpacaFaceDisplay(fullscreen=face_fullscreen)
                if not self.face_display.enabled:
                    print("[WARN] Face display is unavailable (tkinter/GUI missing).")
                    self.face_display = None
                    self.use_face_display = False
                else:
                    self.face_display.start()
                    if self.face_display.last_error:
                        raise RuntimeError(self.face_display.last_error)
                    self._set_face("idle", self._idle_face_subtitle())
                    print("[FACE] Face display started")
            except Exception as e:
                print(f"[WARN] Face display failed to start: {e}")
                self.face_display = None
                self.use_face_display = False

        # Motion hint state for pupil direction.
        self._gaze_from_cmd = 0.0
        self._gaze_from_rollout = 0.0
        self._last_motion_hint_ts = 0.0
        self._gaze_smoothed = 0.0
        self._gaze_latched_sign = 0
        self._gaze_last_sign_change_ts = 0.0
        self._gaze_debug_last_print_ts = 0.0

        if self.use_ros and self.ros_node is not None:
            self.ros_node.set_motion_hint_callback(self._on_motion_hint)
            self.ros_node.set_rollout_preview_callback(self._on_rollout_preview)
            if self._use_face_speaker and self._face_speaker_node is not None:
                # With wake word: face_speaker stays off until wake word is heard (AWAKE state).
                # Without wake word: robot is always "awake", start tracking immediately.
                initial_enabled = not self.use_wake_word
                self._face_speaker_node.set_enabled(initial_enabled)
                if initial_enabled:
                    print("[FACE_SPEAKER] Active-speaker tracking enabled.")
                else:
                    print("[FACE_SPEAKER] Active-speaker tracking ready — activates after wake word.")

    def _set_face(self, expression: str, subtitle: str = "") -> None:
        # When wake-word mode is enabled and not currently awake, keep the prompt visible.
        if self.use_wake_word and self.state != "guiding" and not self._wake_active:
            expression = "idle"
            subtitle = self._wake_prompt_text()
        if self.face_display is not None:
            self.face_display.set_expression(expression, subtitle)

    def _refresh_wake_window(self) -> None:
        """Refresh wake timeout so follow-up turns don't require immediate wake phrase repeats."""
        if not self.use_wake_word:
            return
        self._wake_active = True
        self._last_wake_time = time.time()

    def _append_transcript(self, speaker: str, text: str) -> None:
        line = (text or "").strip()
        if not line:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        try:
            with self._transcript_lock:
                with open(self.transcript_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {speaker}: {line}\n")
        except Exception as e:
            print(f"[LOG] Failed to write transcript: {e}")

    # Whisper hallucinations produced by noise with no real speech.
    _WHISPER_HALLUCINATIONS = {
        "thank you", "thanks for watching", "thank you for watching",
        "thanks for listening", "you", "the", ".", "...", "okay", "ok",
        "subtitles by", "www.", "♪", "music", "applause",
    }

    def _is_noise_transcript(self, text: str) -> bool:
        """Return True if transcript looks like a Whisper hallucination or non-English noise."""
        t = text.strip()
        # Drop if majority of characters are non-ASCII (Japanese, Chinese, Arabic, etc.)
        non_ascii = sum(1 for c in t if ord(c) > 127)
        if len(t) > 0 and non_ascii / len(t) > 0.25:
            print(f"[FILTER] Dropping non-ASCII transcript: {t[:60]!r}")
            return True
        # Drop known Whisper noise hallucinations.
        norm = re.sub(r"[^a-z0-9\s]", "", t.lower()).strip()
        if norm in self._WHISPER_HALLUCINATIONS or not norm:
            print(f"[FILTER] Dropping hallucination: {t[:60]!r}")
            return True
        # Drop very short transcripts that are just punctuation or a single letter.
        if len(norm.replace(" ", "")) < 2:
            print(f"[FILTER] Dropping too-short transcript: {t[:60]!r}")
            return True
        return False

    def _should_enqueue_transcript(self, transcript: str, item_id: Optional[str] = None) -> bool:
        """Return False for duplicate or noisy transcript events."""
        t = (transcript or "").strip()
        if not t:
            return False
        if self._is_noise_transcript(t):
            return False

        if item_id:
            if item_id in self._seen_transcript_item_ids:
                print(f"[DEDUP] Dropped duplicate transcript by item_id={item_id}")
                return False
            self._seen_transcript_item_ids.add(item_id)

        now = time.monotonic()
        norm = re.sub(r"\s+", " ", t.lower()).strip()

        # Drop stale dedupe entries.
        while self._recent_transcripts and (now - self._recent_transcripts[0][0]) > self._transcript_dedupe_window_sec:
            self._recent_transcripts.popleft()

        for _, prev in self._recent_transcripts:
            if prev == norm:
                print("[DEDUP] Dropped duplicate transcript by recent text match")
                return False

        self._recent_transcripts.append((now, norm))
        return True

    def _wake_prompt_text(self) -> str:
        if self.wake_phrase.strip():
            return f'Say "{self.wake_phrase.title()}"'
        return 'Say "Hey Alpaca"'

    def _idle_face_subtitle(self) -> str:
        if not self.use_wake_word:
            return "Ready"
        if not self._wake_active:
            return self._wake_prompt_text()
        return "Ready"

    def _mark_wake_detected_ui(self) -> None:
        # Intentionally disabled per UX decision: no transient wake-detected banner.
        pass

    def _set_face_gaze(self, gaze_x: float) -> None:
        if self.face_display is None:
            return
        try:
            self.face_display.set_gaze(max(-1.0, min(1.0, float(gaze_x))))
        except Exception:
            pass

    def _set_face_rollout_preview(self, points) -> None:
        if self.face_display is None:
            return
        try:
            self.face_display.set_rollout_preview(points)
        except Exception:
            pass

    def _refresh_face_gaze(self) -> None:
        has_fresh_motion = (time.monotonic() - self._last_motion_hint_ts) < 0.6
        active = self.state == "guiding" or self.dummy_motion or has_fresh_motion
        if not active:
            self._gaze_smoothed = 0.0
            self._gaze_latched_sign = 0
            self._set_face_gaze(0.0)
            return

        is_guiding = self.state == "guiding"

        if is_guiding:
            # Nav mode: rollout-dominant blend for anticipatory turn cues.
            raw = 0.90 * self._gaze_from_rollout + 0.10 * self._gaze_from_cmd
            deadband = 0.10
            latch_flip_time = 0.45
            latch_flip_strength = 0.30
            latch_decay_time = 0.35
            alpha = 0.22
        else:
            # Face-speaker tracking: cmd_vel is the sole signal; keep proportional.
            # cmd gaze tops out at max_angular_z/omega_ref ≈ 0.625 so use the full value.
            raw = self._gaze_from_cmd
            deadband = 0.04
            latch_flip_time = 0.20   # snappier direction changes for face tracking
            latch_flip_strength = 0.15
            latch_decay_time = 0.18
            alpha = 0.42             # faster response, less lag

        if abs(raw) < deadband:
            raw = 0.0

        now = time.monotonic()
        sign = 0
        if raw > 1e-6:
            sign = 1
        elif raw < -1e-6:
            sign = -1

        # Latch direction briefly to avoid back-and-forth jitter.
        if sign != 0:
            if self._gaze_latched_sign == 0:
                self._gaze_latched_sign = sign
                self._gaze_last_sign_change_ts = now
            elif sign != self._gaze_latched_sign:
                if abs(raw) >= latch_flip_strength or (now - self._gaze_last_sign_change_ts) >= latch_flip_time:
                    self._gaze_latched_sign = sign
                    self._gaze_last_sign_change_ts = now
                else:
                    sign = self._gaze_latched_sign
        else:
            if (now - self._gaze_last_sign_change_ts) >= latch_decay_time:
                self._gaze_latched_sign = 0

        if is_guiding and sign != 0:
            # Nav mode: boost to a strongly readable minimum so subtle rollouts still
            # produce a visible directional cue.
            if abs(raw) < 0.20:
                target = 0.0
            else:
                target = sign * max(abs(raw), 0.48)
        else:
            # Face-speaker mode: proportional — mirror the P-controller output directly.
            target = raw

        self._gaze_smoothed += alpha * (target - self._gaze_smoothed)
        self._set_face_gaze(self._gaze_smoothed)

    def _on_motion_hint(self, source: str, value: float):
        v = max(-1.0, min(1.0, float(value)))
        if source == "rollout":
            self._gaze_from_rollout = v
        else:
            self._gaze_from_cmd = v
        self._last_motion_hint_ts = time.monotonic()
        # Rate-limited debug so we can confirm real ROS hints are flowing.
        if (self._last_motion_hint_ts - self._gaze_debug_last_print_ts) > 1.0:
            self._gaze_debug_last_print_ts = self._last_motion_hint_ts
            # print(f"[GAZE] src={source} rollout={self._gaze_from_rollout:.2f} cmd={self._gaze_from_cmd:.2f}")
        self._refresh_face_gaze()

    def _on_rollout_preview(self, points):
        self._set_face_rollout_preview(points)

    def shutdown(self) -> None:
        if self.face_display is not None:
            self.face_display.close()
            self.face_display = None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: RobotState, reason: str = "") -> None:
        """Transition to a new RobotState, applying all side-effects atomically.

        Side-effects:
        - Keeps self.state ("speaking"|"guiding") in sync for backward compat.
        - Enables/disables face_speaker (off during GUIDING, on otherwise).
        - Prints an FSM log line.
        """
        old = self.robot_state
        if old == new_state:
            return
        self.robot_state = new_state
        # Backward-compat string state kept in sync
        self.state = "guiding" if new_state == RobotState.GUIDING else "speaking" 
        tag = f"[FSM] {old.value} → {new_state.value}"
        if reason:
            tag += f"  ({reason})"
        print(tag)
        # Gate face_speaker based on mode:
        #   wake_word ON  → only AWAKE/ARRIVED; robot should track only when explicitly woken
        #   wake_word OFF → all states except GUIDING; robot always tracks unless navigating
        if self._use_face_speaker and self._face_speaker_node is not None:
            if self.use_wake_word:
                fs_on = new_state in (RobotState.AWAKE, RobotState.ARRIVED)
            else:
                fs_on = new_state in (RobotState.AWAKE, RobotState.ARRIVED)
                # fs_on = new_state != RobotState.GUIDING
            self._face_speaker_node.set_enabled(fs_on)

    def _reset_dialog_state(self) -> None:
        self.dialog_state = "idle"
        self.pending_location = "na"

    def _apply_dialog_state_from_output(self, out: AssistantOutput) -> None:
        pending = out.pending_location.value if isinstance(out.pending_location, Enum) else out.pending_location
        self.dialog_state = out.dialog_state
        self.pending_location = pending or "na"

    def _cancel_current_navigation(self, reason: str = "Okay, stopping.") -> None:
        """Stop the current nav2 goal (if any) and switch back to speaking mode."""
        if self.use_ros and self.ros_node is not None:
            self.ros_node.cancel_navigation()

        self._nav_cancel_epoch += 1
        self._refresh_wake_window()
        self._transition_to(RobotState.AWAKE, "navigation canceled")
        self._current_navigation_location = None
        self._reset_dialog_state()
        self._set_face("speaking", "Navigation canceled")
        self._set_face_gaze(0.0)

        print("[NAV] Navigation canceled.")
        self.speak_response(reason)

    async def _cancel_current_navigation_async(self, reason: str = "Okay, stopping.") -> None:
        """Async-safe cancel path for websocket/event-loop handlers."""
        if self.use_ros and self.ros_node is not None:
            self.ros_node.cancel_navigation()

        self._nav_cancel_epoch += 1
        self._refresh_wake_window()
        self._transition_to(RobotState.AWAKE, "navigation canceled")
        self._current_navigation_location = None
        self._reset_dialog_state()
        self._set_face("speaking", "Navigation canceled")
        self._set_face_gaze(0.0)

        print("[NAV] Navigation canceled.")
        await asyncio.to_thread(self.speak_response, reason)

    def _normalize_text(self, s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _similarity(self, a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()

    def _build_wake_variants(self, wake_phrase: str) -> List[str]:
        base = self._normalize_text(wake_phrase)
        parts = base.split()
        variants = {base}

        if base:
            variants.add(base.replace(" ", ""))  # e.g. "heyalpaca"

        if parts:
            keyword = parts[-1]
            variants.add(keyword)  # e.g. "alpaca"
            if len(keyword) >= 4:
                variants.add(f"{keyword[:2]} {keyword[2:]}")  # e.g. "al paca"

                # Common STT variations for accents/noise around "alpaca"
                if keyword == "alpaca":
                    variants.update({
                        "alpaka", "alpacca", "alpacka",
                        "al paca", "al paka", "alpacar", "alpaga",
                        "alpacca", "alpacker", "alpacaa"
                    })
                    variants.update({
                        f"hey {keyword}", "hey al paca", "hey al paka", "hay alpaca",
                        "hey alpaga", "hi alpaca", "hello alpaca", "okay alpaca",
                        "yo alpaca"
                    })

        # Keep only non-empty variants
        return [v for v in variants if v]

    def _check_wake_word(self, transcript: str) -> tuple[bool, str]:
        """
        Check if transcript contains the wake phrase and update wake state.
        Returns (should_process, cleaned_transcript)
        """
        import time
        now = time.time()
        if self.state == "guiding":
            return True, transcript
        normalized = self._normalize_text(transcript)
        tokens = normalized.split()

        # Ignore obvious noise unless we get a very strong wake-word hit
        if len(tokens) < 1:
            return False, transcript

        # Debounce repeated triggers from echo/noise
        if (now - self._last_wake_trigger_time) < self._wake_debounce_sec:
            if self._wake_active and (now - self._last_wake_time) < self.wake_timeout:
                return True, transcript
            return False, transcript

        max_n = max((len(v.split()) for v in self._wake_variants), default=1)
        best_score = 0.0
        best_i = None
        best_n = 0

        for n in range(1, max_n + 1):
            for i in range(0, max(0, len(tokens) - n + 1)):
                cand = " ".join(tokens[i:i + n])
                for target in self._wake_variants:
                    score = self._similarity(cand, target)
                    if score > best_score:
                        best_score = score
                        best_i = i
                        best_n = n

        at_start = best_i is not None and best_i <= self._wake_start_window_tokens
        detected = (best_score >= self._wake_threshold) or (
            at_start and best_score >= self._wake_threshold_start
        )
        # Extra lenient path: if the keyword is heard near start, accept.
        if not detected and tokens:
            kw = self.wake_phrase.split()[-1] if self.wake_phrase.split() else "alpaca"
            first_window = " ".join(tokens[: self._wake_start_window_tokens + 1])
            if kw in first_window or "al paca" in first_window or "alpaka" in first_window:
                detected = True

        if detected and (self._wake_active or at_start):
            self._wake_active = True
            self._last_wake_time = now
            self._last_wake_trigger_time = now
            print(f"[WAKEWORD] Wake phrase detected (score={best_score:.1f})! Listening for {self.wake_timeout}s...")
            if self.robot_state != RobotState.GUIDING:
                self._transition_to(RobotState.AWAKE, "wake word detected")

            cleaned_tokens = tokens
            if best_i is not None and best_n > 0:
                cleaned_tokens = tokens[:best_i] + tokens[best_i + best_n:]
            cleaned = " ".join(cleaned_tokens).strip()
            cleaned = re.sub(r"^(hey|hi|hello)\s*", "", cleaned, flags=re.IGNORECASE).strip()

            return True, cleaned

        # Check if still within wake timeout
        if self._wake_active:
            if now - self._last_wake_time < self.wake_timeout:
                return True, transcript
            else:
                self._wake_active = False
                print(f"[WAKEWORD] Wake timeout expired. Say '{self.wake_phrase}' to re-activate.")
                self._transition_to(RobotState.IDLE, "wake timeout")
                self._set_face("idle", self._idle_face_subtitle())
                return False, transcript

        # Log ignored transcripts for debugging
        print(f"[WAKEWORD] Not awake, ignoring: '{transcript[:60]}...' (best_score={best_score:.1f})")
        return False, transcript



    def _on_goal_reached(self, location_name: Optional[str]):
        """
        Callback invoked when the robot reaches the navigation goal.
        Announces arrival to the user and switches back to speaking mode.
        """
        print(f"[NAV] Goal reached callback triggered for location: {location_name}")
        
        if self.state != "guiding":
            print("[NAV] Not in guiding mode, ignoring goal reached callback")
            return
        
        # Transition to ARRIVED — face_speaker re-enables here
        self._nav_cancel_epoch += 1
        self._transition_to(RobotState.ARRIVED, "goal reached")
        self._reset_dialog_state()
        self._set_face("arrived", f"Arrived: {location_name}" if location_name else "Arrived")
        self._set_face_gaze(0.0)

        # Announce arrival to the user via TTS
        if location_name:
            announcement = f"We have reached {location_name}."
        else:
            announcement = "We have reached the destination."

        print(f"[TTS] Announcing: {announcement}")
        self._current_navigation_location = None
        self.speak_response(announcement)

        self._transition_to(RobotState.IDLE, "arrival announced")

    # NOTE: Legacy/simple wake-word implementation intentionally commented out.
    # The active implementation above uses fuzzy matching + debounce + thresholds.
    #
    # def _check_wake_word(self, transcript: str) -> tuple[bool, str]:
    #     import time
    #     import re
    #     transcript_lower = transcript.lower()
    #     wake_variations = [self.wake_phrase, self.wake_phrase.replace(" ", "")]
    #     key_words = self.wake_phrase.split()
    #     if len(key_words) > 1:
    #         wake_variations.append(key_words[-1])
    #         wake_variations.append(key_words[-1][:2] + " " + key_words[-1][2:])
    #     detected_phrase = None
    #     for variation in wake_variations:
    #         if variation in transcript_lower:
    #             detected_phrase = variation
    #             break
    #     if detected_phrase:
    #         self._wake_active = True
    #         self._last_wake_time = time.time()
    #         cleaned = re.sub(re.escape(detected_phrase), "", transcript_lower, flags=re.IGNORECASE).strip()
    #         cleaned = re.sub(r"^(hey|hi|hello)\s*", "", cleaned, flags=re.IGNORECASE).strip()
    #         return True, cleaned
    #     if self._wake_active and (time.time() - self._last_wake_time < self.wake_timeout):
    #         return True, transcript
    #     self._wake_active = False
    #     return False, transcript



    def _have_pulse_tools(self) -> bool:
        import shutil, os
        return (shutil.which("parec") is not None and
                shutil.which("pacat") is not None and
                (os.path.exists("/tmp/pulse/native") or ("PULSE_SERVER" in os.environ)))

    def _pick_backend(self) -> str:
        if self.audio_backend != "auto":
            return self.audio_backend
        return "pulse" if self._have_pulse_tools() else "alsa"

    def _resolve_sd_device(self, spec, kind: str):
        """
        spec can be:
        - None  -> use sd.default.device[0/1]
        - int / digit-string -> device index
        - substring -> first device containing substring (case-insensitive)
        kind: "input" or "output"
        """
        if spec is None:
            return sd.default.device[0] if kind == "input" else sd.default.device[1]

        if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
            return int(spec)

        # substring match
        name_sub = str(spec).lower()
        for i, d in enumerate(sd.query_devices()):
            if kind == "input" and d.get("max_input_channels", 0) <= 0:
                continue
            if kind == "output" and d.get("max_output_channels", 0) <= 0:
                continue
            if name_sub in d.get("name", "").lower():
                return i

        raise RuntimeError(f"No sounddevice {kind} device matched '{spec}'.")

    def _build_static_system_msg(self) -> str:
        """Built once at init. Byte-identical across all turns so OpenAI auto-caches the prefix."""
        loc_info = self.task_info.get("location_info", {})
        # Strip coordinates/yaw — LLM only needs names + tags for intent matching.
        llm_loc_info = {name: {"tags": data.get("tags", [])} for name, data in loc_info.items()}
        return (
            f"{self.system_prompt}\n\n"
            "TASK_INFO_JSON:\n"
            f"{json.dumps({'location_info': llm_loc_info}, indent=2)}\n\n"
            "CONVERSATIONAL_STYLE:\n"
            "- Sound like a friendly in-person guide, not a form.\n"
            "- Use natural contractions and varied wording.\n"
            "- Ask one short follow-up question when helpful to keep the flow.\n"
            "- Avoid repeating the same opener across consecutive turns."
        )

    def _build_dynamic_state_msg(self) -> str:
        """Built each turn — robot + dialog state only. Kept short to minimise non-cached tokens."""
        msg = (
            "ROBOT_STATE_JSON:\n"
            f"{json.dumps({'state': self.state, 'current_navigation_location': self._current_navigation_location}, indent=2)}\n\n"
            "DIALOG_STATE_JSON:\n"
            f"{json.dumps({'dialog_state': self.dialog_state, 'pending_location': self.pending_location}, indent=2)}"
        )
        if self.state == "guiding":
            msg += (
                "\n\nWhile guiding:\n"
                "- You are acting as a tour guide in the building.\n"
                "- Use web search for researcher/lab info, building history, or time-sensitive facts.\n"
                "- Keep responses concise (1-3 sentences). Do not fabricate."
            )
        return msg

    def build_system_message(self) -> str:
        return self._static_system_msg + "\n\n" + self._build_dynamic_state_msg()
        
    def speak_response(self, text: str) -> None:
        """Convert text to speech using OpenAI TTS streaming and play audio."""
        text = sanitize_for_tts(text)
        if not text.strip():
            return
        with self._tts_lock:
            self._append_transcript("BOT", text)
            try:
                print(f"[TTS] Speaking: {text[:50]}...")
                self._tts_playing = True

                def _on_audio_start():
                    self._set_face("speaking", "Speaking...")

                backend = getattr(self, "_chosen_backend", None)
                with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1-hd",
                    voice="nova",
                    input=text,
                    response_format="pcm",
                ) as response:
                    if backend == "alsa":
                        if self.sd_output_device is not None:
                            sd.default.device = (sd.default.device[0], self._resolve_sd_device(self.sd_output_device, "output"))
                        self._stream_pcm_via_sounddevice(response, on_start=_on_audio_start)
                    else:
                        self._stream_pcm_via_pulse(response, sr=24000, channels=1, sink=self.pulse_sink, on_start=_on_audio_start)

            except Exception as e:
                print(f"[TTS] Error: {e}")
            finally:
                # Wait for speaker echo to decay before re-enabling STT.
                time.sleep(0.4)
                # Drain mic chunks that arrived during TTS + cooldown.
                self._clear_audio_queue()
                # Tell the OpenAI server to discard any audio already buffered.
                if getattr(self, "_loop", None) is not None:
                    self._loop.call_soon_threadsafe(self._audio_q.put_nowait, None)
                self._tts_playing = False
                self._refresh_wake_window()
                if self.state == "guiding":
                    self._set_face("guiding", f"Guiding to {self._current_navigation_location or 'destination'}")
                else:
                    self._set_face("idle", self._idle_face_subtitle())

    def _stream_pcm_via_pulse(self, response, sr: int = 24000, channels: int = 1, sink: str | None = None, on_start=None) -> None:
        """Stream TTS PCM16 chunks directly into pacat for low-latency playback."""
        if shutil.which("pacat") is None:
            raise RuntimeError("pacat not found (install pulseaudio-utils).")

        cmd = [
            "pacat", "--playback", "--raw", "--format=s16le",
            f"--rate={sr}", f"--channels={channels}",
        ]
        sink_name = sink or os.environ.get("PULSE_SINK")
        if sink_name:
            cmd.append(f"--device={sink_name}")

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            first = True
            for chunk in response.iter_bytes(chunk_size=4096):
                if first and on_start is not None:
                    on_start()
                    first = False
                assert p.stdin is not None
                p.stdin.write(chunk)
            p.stdin.close()
            try:
                rc = p.wait(timeout=60)
                if rc != 0:
                    raise RuntimeError(f"pacat exited with code {rc}")
            except subprocess.TimeoutExpired:
                print("[TTS] pacat timed out — killing process")
                p.kill()
                p.wait()
        finally:
            try:
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
            except Exception:
                pass

    def _stream_pcm_via_sounddevice(self, response, on_start=None) -> None:
        """Stream TTS PCM16 chunks to sounddevice output for low-latency playback."""
        buf = b""
        stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
        stream.start()
        try:
            first = True
            for chunk in response.iter_bytes(chunk_size=4096):
                if first and on_start is not None:
                    on_start()
                    first = False
                buf += chunk
                usable = len(buf) - (len(buf) % 2)
                if usable > 0:
                    stream.write(np.frombuffer(buf[:usable], dtype=np.int16))
                    buf = buf[usable:]
            if len(buf) >= 2:
                usable = len(buf) - (len(buf) % 2)
                stream.write(np.frombuffer(buf[:usable], dtype=np.int16))
            stream.stop()
        finally:
            stream.close()

    # def _mic_callback(self, indata, frames, time, status):
        # if status:
        #     # You could log this
        #     pass

        # # Skip audio capture while TTS is playing to prevent feedback
        # if self._tts_playing:
        #     return

        # # IMPORTANT: sounddevice calls this on a separate audio thread.
        # # asyncio.Queue is NOT thread-safe; schedule the put on the event loop.
        # if getattr(self, "_loop", None) is not None:
        #     chunk = indata.tobytes()
        #     self._loop.call_soon_threadsafe(self._audio_q.put_nowait, chunk)

    def _mic_callback(self, indata, frames, time, status):
        # if status:
        #     # don't spam; but you can print(status) if needed
        #     pass

        # if self._tts_playing:
        #     return

        # # indata is int16 if dtype="int16"
        # x = indata.reshape(-1).astype(np.int16)

        # # quick audio level check
        # peak = int(np.max(np.abs(x))) if x.size else 0
        # rms = float(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2))) if x.size else 0.0

        # now = time.inputBufferAdcTime if hasattr(time, "inputBufferAdcTime") else None
        # # fallback to loop time
        # loop = self._loop
        # tnow = loop.time() if loop is not None else 0.0

        # if loop is not None and (tnow - self._dbg_last_audio_print) > 1.0:
        #     self._dbg_last_audio_print = tnow
        #     print(f"[AUDIO] peak={peak} rms={rms:.4f} frames={frames}")

        # if self._loop is not None:
        #     chunk = indata.tobytes()
        #     self._loop.call_soon_threadsafe(self._audio_q.put_nowait, chunk)

        if self._tts_playing:
            return
        if self._loop is None:
            return

        x = indata.reshape(-1).astype(np.int16)

        # Debug level meter (optional)
        peak = int(np.max(np.abs(x))) if x.size else 0
        rms = float(np.sqrt(np.mean((x.astype(np.float32) / 32768.0) ** 2))) if x.size else 0.0
        tnow = self._loop.time()
        if (tnow - getattr(self, "_dbg_last_audio_print", 0.0)) > 1.0:
            self._dbg_last_audio_print = tnow
            # print(f"[AUDIO] peak={peak} rms={rms:.4f} frames={frames} mic_sr={self._mic_sr}")

        # High-pass filter before resample (removes low-freq mechanical noise)
        x = self._apply_noise_filter(x, self._mic_sr)
        # Resample to 24k for the API
        y = pcm16_resample_linear(x, self._mic_sr, 24000)

        self._loop.call_soon_threadsafe(self._audio_q.put_nowait, y.tobytes())

    
    def _apply_noise_filter(self, x: np.ndarray, sr: int) -> np.ndarray:
        """High-pass filter to cut low-frequency mechanical noise (e.g. lidar motor hum).
        Only active when --noise-filter-hz > 0.  Uses scipy butter IIR (4th-order).
        Falls back to no-op if scipy is unavailable."""
        if self.noise_filter_hz <= 0:
            return x
        try:
            from scipy.signal import butter, sosfilt
            if self._hpf_sos is None or self._hpf_sos_sr != sr:
                self._hpf_sos = butter(4, self.noise_filter_hz, btype='high', fs=sr, output='sos')
                self._hpf_sos_sr = sr
            filtered = sosfilt(self._hpf_sos, x.astype(np.float32))
            return np.clip(np.round(filtered), -32768, 32767).astype(np.int16)
        except ImportError:
            return x  # scipy not available — skip filter

    def _clear_audio_queue(self):
        """Clear any buffered audio after TTS playback."""
        cleared = 0
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared > 0:
            print(f"[STT] Cleared {cleared} audio chunks from buffer")

    async def _pulse_mic_reader(self, source: str | None = None, rate: int = 44100, channels: int = 2):
        """
        Capture mic audio from PulseAudio using `parec` and push 24k mono PCM16 chunks to self._audio_q.
        Requires: pulseaudio-utils (provides `parec`).
        """
        cmd = [
            "parec",
            "--raw",
            f"--format=s16le",
            f"--rate={rate}",
            f"--channels={channels}",
        ]
        if source:
            cmd.append(f"--device={source}")

        # Start subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        bytes_per_sample = 2  # s16le
        bytes_per_frame = bytes_per_sample * channels

        # read ~20ms chunks (or your self.frame_ms)
        frame_ms = getattr(self, "frame_ms", 20)
        frames_per_chunk = int(rate * frame_ms / 1000)
        chunk_bytes = frames_per_chunk * bytes_per_frame

        print(f"[AUDIO] Pulse capture: rate={rate} ch={channels} chunk_bytes={chunk_bytes}")
        if source:
            print(f"[AUDIO] Pulse source: {source}")

        try:
            while True:
                data = await proc.stdout.readexactly(chunk_bytes)

                if self._tts_playing:
                    continue

                x = np.frombuffer(data, dtype=np.int16)
                if channels == 2:
                    x = x.reshape(-1, 2)
                    mono = x.mean(axis=1).astype(np.int16)
                else:
                    mono = x

                # Debug level meter
                peak = int(np.max(np.abs(mono))) if mono.size else 0
                rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2))) if mono.size else 0.0
                tnow = self._loop.time()
                if (tnow - getattr(self, "_dbg_last_audio_print", 0.0)) > 1.0:
                    self._dbg_last_audio_print = tnow
                    print(f"[AUDIO] peak={peak} rms={rms:.4f} (pulse)")

                # High-pass filter before resample (removes low-freq mechanical noise)
                mono = self._apply_noise_filter(mono, rate)
                # Resample to 24k for Realtime (audio/pcm supports 24k only)
                y = pcm16_resample_linear(mono, rate, 24000)
                self._audio_q.put_nowait(y.tobytes())
        except asyncio.IncompleteReadError:
            pass
        finally:
            proc.terminate()
            try:
                await proc.wait()
            except Exception:
                pass

    def _play_pcm_via_pulse(self, pcm_bytes: bytes, sr: int = 24000, channels: int = 1, sink: str | None = None) -> None:
        """
        Play raw PCM16LE bytes via PulseAudio using pacat.
        """
        import shutil
        import subprocess
        import os

        if shutil.which("pacat") is None:
            raise RuntimeError("pacat not found (install pulseaudio-utils).")

        cmd = [
            "pacat",
            "--playback",
            "--raw",
            "--format=s16le",
            f"--rate={sr}",
            f"--channels={channels}",
        ]

        sink_name = sink or os.environ.get("PULSE_SINK")
        if sink_name:
            cmd.append(f"--device={sink_name}")

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            assert p.stdin is not None
            p.stdin.write(pcm_bytes)
            p.stdin.close()
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"pacat exited with code {rc}")
        finally:
            try:
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
            except Exception:
                pass

    async def start(self):
        # GA streaming transcription: connect with intent=transcription
        ws_url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            # GA: do NOT send OpenAI-Beta: realtime=v1
        }

        # sounddevice invokes callbacks on a separate audio thread; keep a handle to the asyncio loop
        self._loop = asyncio.get_running_loop()

        # websockets header kw changed across versions (extra_headers -> additional_headers)
        try:
            ws_cm = websockets.connect(ws_url, additional_headers=headers, max_size=2**23)
        except TypeError:
            ws_cm = websockets.connect(ws_url, extra_headers=headers, max_size=2**23)

        async with ws_cm as ws:
            # Wait for session.created before configuring
            initial_msg = await ws.recv()
            initial_evt = json.loads(initial_msg)
            print(f"Initial event: {initial_evt.get('type')}")

            # GA transcription configuration uses session.update (not transcription_session.update).
            config_msg = {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "noise_reduction": {"type": "near_field"},
                            "transcription": {
                                "model": self.transcription_model,  # gpt-4o-transcribe or gpt-4o-mini-transcribe
                                "language": "en",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": self.vad_threshold,
                                "prefix_padding_ms": 500,
                                "silence_duration_ms": 800,
                                # Transcription-only: don't auto-generate assistant responses
                                "create_response": False,
                            },
                        }
                    }
                },
            }
            print(f"Sending config: {json.dumps(config_msg, indent=2)}")
            await ws.send(json.dumps(config_msg))

            # Use the Yeti pulse source (from `pactl list short sources`)
            # yeti_source = "alsa_input.usb-Blue_Microphones_Yeti_Stereo_Microphone_REV8-00.analog-stereo"

            backend = self._pick_backend()
            print(f"[AUDIO] backend={backend}")

            pulse_task = None
            stream = None

            if backend == "pulse":
                # Source priority: CLI arg -> system default (None = parec uses PulseAudio default source)
                src = self.pulse_source  # None means use system default
                pulse_task = asyncio.create_task(self._pulse_mic_reader(source=src, rate=44100, channels=2))
            else:
                # sounddevice path
                input_dev = self._resolve_sd_device(self.sd_input_device, kind="input")  # helper below
                dev_info = sd.query_devices(input_dev)
                self._mic_sr = int(dev_info["default_samplerate"])
                print(f"[AUDIO] Using input device {input_dev}: {dev_info['name']} @ {self._mic_sr} Hz")

                stream = sd.InputStream(
                    device=input_dev,
                    samplerate=self._mic_sr,
                    channels=1,
                    dtype="int16",
                    blocksize=int(self._mic_sr * self.frame_ms / 1000),
                    callback=self._mic_callback,  # ensure this puts bytes into self._audio_q
                )
                stream.start()

            self._loop = asyncio.get_running_loop()
            self._dbg_last_audio_print = 0.0

            tasks = []
            if pulse_task is not None:
                tasks.append(pulse_task)
            tasks.extend([
                asyncio.create_task(self._audio_sender(ws)),
                asyncio.create_task(self._event_receiver(ws)),
                asyncio.create_task(self._utterance_worker()),
                asyncio.create_task(self._gaze_watchdog_loop()),
            ])
            if self.dummy_motion:
                print("[TEST] Dummy motion hints enabled (synthetic rollout + cmd_vel).")
                tasks.append(asyncio.create_task(self._dummy_motion_loop()))

            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

                if stream is not None:
                    stream.stop()
                    stream.close()

    async def _dummy_motion_loop(self):
        """Generate synthetic anticipatory rollout/cmd turn hints for off-robot testing."""
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            rollout_hint = float(np.sin(0.55 * t))
            cmd_hint = float(0.9 * np.sin(0.55 * t + 0.60))
            self._on_motion_hint("rollout", rollout_hint)
            self._on_motion_hint("cmd", cmd_hint)
            # Synthetic rollout path preview (starts center-bottom, bends with rollout_hint).
            pts = []
            for i in range(16):
                s = i / 15.0
                x = 0.5 + 0.36 * rollout_hint * (s ** 1.4)
                y = 0.1 + 0.8 * s
                pts.append((float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))))
            self._set_face_rollout_preview(pts)
            await asyncio.sleep(0.1)

    async def _gaze_watchdog_loop(self):
        """Recenters gaze if motion hints become stale."""
        while True:
            if not self.dummy_motion and self._last_motion_hint_ts > 0.0:
                if (time.monotonic() - self._last_motion_hint_ts) > 0.7:
                    self._gaze_from_cmd *= 0.7
                    self._gaze_from_rollout *= 0.7
                    if abs(self._gaze_from_cmd) < 0.05:
                        self._gaze_from_cmd = 0.0
                    if abs(self._gaze_from_rollout) < 0.05:
                        self._gaze_from_rollout = 0.0
                    self._refresh_face_gaze()
            await asyncio.sleep(0.1)


    async def _audio_sender(self, ws):
        # Continually send audio frames to input_audio_buffer.append
        while True:
            chunk = await self._audio_q.get()
            # None sentinel: flush accumulated audio on the server side.
            # Processed even during TTS so the clear reaches the server promptly.
            if chunk is None:
                await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                print("[STT] Server audio buffer cleared")
                continue
            # Skip sending if TTS is playing
            if self._tts_playing:
                continue
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": b64encode_audio_pcm16(chunk),
                    }
                )
            )

    async def _event_receiver(self, ws):
        # Listen for:
        # - VAD events: speech_started / speech_stopped :contentReference[oaicite:8]{index=8}
        # - committed item IDs (ordering)
        # - transcription completed events with final transcript text :contentReference[oaicite:9]{index=9}
        async for raw in ws:
            evt = json.loads(raw)
            t = evt.get("type")
            print(t)  # Log all event types for debugging

            # Handle error events with full details
            if t == "error":
                error_info = evt.get("error", {})
                print(f"[ERROR] Code: {error_info.get('code')}")
                print(f"[ERROR] Message: {error_info.get('message')}")
                print(f"[ERROR] Full event: {json.dumps(evt, indent=2)}")
                continue

            elif t == "session.updated":
                print(f"[SESSION] Updated: {json.dumps(evt.get('session', {}), indent=2)}")

            elif t == "transcription_session.updated":
                print("[SESSION] Transcription session configured successfully")

            elif t == "input_audio_buffer.speech_started":
                print("… user speaking …")
                if self.state == "guiding":
                    self._set_face("listening", "Listening while guiding")
                elif not self.use_wake_word or self._wake_active:
                    self._set_face("listening", "Listening...")
                # Pre-enable face_speaker as soon as someone starts speaking so
                # the robot is already turning by the time the wake word is confirmed.
                if (self.use_wake_word
                        and self.robot_state == RobotState.IDLE
                        and self._use_face_speaker
                        and self._face_speaker_node is not None):
                    self._face_speaker_node.set_enabled(True)

            elif t == "input_audio_buffer.speech_stopped":
                print("… speech stopped (VAD) …")
                if not self.use_wake_word or self._wake_active:
                    self._set_face("thinking", "Thinking...")

            elif t == "input_audio_buffer.committed":
                # In VAD mode, the API commits audio chunks and returns item_id
                self._latest_committed_item_id = evt.get("item_id")

            elif t == "conversation.item.input_audio_transcription.completed":
                # Final transcript for an item_id
                item_id = evt.get("item_id")
                transcript = evt.get("transcript", "").strip()
                if not transcript:
                    continue

                if self.state == "guiding":
                    # Allow interrupt commands while guiding
                    tl = transcript.lower()
                    if any(k in tl for k in [
                        "stop navigation", "cancel navigation", "abort navigation", "stop moving",
                        "stop", "wait", "hold on", "never mind", "nevermind", "cancel", "abort", "halt", "pause",
                    ]):
                        await self._cancel_current_navigation_async("Okay, stopping navigation. What would you like to do?")
                        continue
                    # Otherwise allow small talk / questions while guiding (do not ignore)


                # Best-effort: only process the latest committed item
                if self._latest_committed_item_id and item_id != self._latest_committed_item_id:
                    continue

                if self.use_wake_word and self.state != "guiding":  # If guiding, allow all speech through (including without wake word)
                    should_process, transcript = self._check_wake_word(transcript)
                    if not should_process:
                        # Undo the pre-emptive face_speaker enable from speech_started.
                        if (self.robot_state == RobotState.IDLE
                                and self._use_face_speaker
                                and self._face_speaker_node is not None):
                            self._face_speaker_node.set_enabled(False)
                        print(f"[WAKEWORD] Ignoring (not awake): {transcript[:50]}...")
                        continue
                    if not transcript:
                        continue  # Wake phrase only, no command

                if not self._should_enqueue_transcript(transcript, item_id=item_id):
                    continue

                print(f"\nTranscript: {transcript}\n")
                await self._utterance_q.put(transcript)

            # Ignore transcription.done to avoid duplicate processing with
            # conversation.item.input_audio_transcription.completed.
            elif t == "transcription.done":
                continue

            # Handle conversation.item.created - check for transcript content
            elif t == "conversation.item.created":
                item = evt.get("item", {})
                item_id = item.get("id")
                # Check if this item contains transcript data
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "input_audio" and c.get("transcript"):
                        transcript = c.get("transcript", "").strip()
                        if transcript:
                            # Check wake word (if enabled)
                            if self.use_wake_word and self.state != "guiding":  # If guiding, allow all speech through (including without wake word)
                                should_process, transcript = self._check_wake_word(transcript)
                                if not should_process:
                                    print(f"[WAKEWORD] Ignoring (not awake): {transcript[:50]}...")
                                    break
                                if not transcript:
                                    break
                            if not self._should_enqueue_transcript(transcript, item_id=item_id):
                                break
                            print(f"\nTranscript: {transcript}\n")
                            await self._utterance_q.put(transcript)
                            break
                # Also log the full item for debugging
                print(f"[DEBUG] conversation.item.created: {json.dumps(item, indent=2)[:500]}")

    async def _utterance_worker(self):
        while True:
            user_text = await self._utterance_q.get()
            try:
                await self._process_utterance(user_text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[ERR] _utterance_worker unhandled exception: {e}")
                if self.state == "guiding":
                    self._set_face("guiding", f"Guiding to {self._current_navigation_location or 'destination'}")
                else:
                    self._set_face("idle", self._idle_face_subtitle())

    async def _process_utterance(self, user_text: str) -> None:
        # If latest_only mode, drain the queue and only keep the most recent
        if self.latest_only:
            skipped = 0
            while not self._utterance_q.empty():
                try:
                    user_text = self._utterance_q.get_nowait()
                    skipped += 1
                except asyncio.QueueEmpty:
                    break
            if skipped > 0:
                print(f"[UTTERANCE] Skipped {skipped} stale utterance(s), processing latest only")
        self._append_transcript("HUMAN", user_text)

        self._set_face("thinking", "Thinking...")
        # Snapshot epoch before LLM call so we can detect cancel/arrival mid-call.
        epoch_before = self._nav_cancel_epoch
        # Run blocking API call off the asyncio loop to avoid websocket ping timeouts.
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(self.generate_structured_response, user_text),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            print("[ERR] LLM call timed out after 25s — dropping utterance.")
            if self.state == "guiding":
                self._set_face("guiding", f"Guiding to {self._current_navigation_location or 'destination'}")
            else:
                self._set_face("idle", self._idle_face_subtitle())
            return
        nav_was_canceled = self._nav_cancel_epoch != epoch_before
        if nav_was_canceled:
            print("[NAV] Cancel/arrival detected during LLM call — sanitizing stale response.")
            if out.next_action == "navigate":
                # Suppress re-navigation to same spot after cancel.
                print("[NAV] Dropping stale navigate action from pre-cancel LLM response.")
                if self.state == "guiding":
                    self._set_face("guiding", f"Guiding to {self._current_navigation_location or 'destination'}")
                else:
                    self._set_face("idle", self._idle_face_subtitle())
                return
            if out.dialog_state == "navigating":
                out.dialog_state = "idle"
                out.pending_location = "na"
        if not out.being_talked_to:
            print("[INFO] User not talking to the robot; ignoring utterance.")
            if self.state == "guiding":
                self._set_face("guiding", f"Guiding to {self._current_navigation_location or 'destination'}")
            else:
                self._set_face("idle", self._idle_face_subtitle())
            return
        print("Assistant JSON:")
        print(out.model_dump_json(indent=2))
        print()
        self._apply_dialog_state_from_output(out)

        # If we're already guiding to this same destination, suppress repeated
        # "I'll guide you..." confirmations for smoother interaction.
        suppress_repeat_guiding_reply = False
        if out.next_action == "navigate":
            out_location = out.location.value if isinstance(out.location, Enum) else out.location
            if self.state == "guiding" and self._current_navigation_location == out_location:
                suppress_repeat_guiding_reply = True
                print(f"[NAV] Already guiding to '{out_location}', suppressing repeated navigation reply.")

        # Speak the response to the user
        # TTS playback is blocking; run in a worker thread.
        if not suppress_repeat_guiding_reply:
            await asyncio.to_thread(self.speak_response, out.answer)

        # Handle navigation state transitions
        self.handle_navigation_response(out)

    async def start_text_mode(self):
        """
        Text input mode for debugging.
        Instead of streaming audio, takes text input from the console.
        """
        print("\n" + "=" * 50)
        print("TEXT INPUT MODE (for debugging)")
        print("Type your messages and press Enter.")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 50 + "\n")

        while True:
            try:
                user_text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
            except EOFError:
                print("\nExiting text mode.")
                break

            if not user_text:
                continue

            if user_text.lower() in ("quit", "exit"):
                print("Exiting text mode.")
                break
            self._append_transcript("HUMAN", user_text)

            out = await asyncio.to_thread(self.generate_structured_response, user_text)
            print("\nAssistant JSON:")
            print(out.model_dump_json(indent=2))
            print()
            self._apply_dialog_state_from_output(out)
            
            # Speak the response to the user
            await asyncio.to_thread(self.speak_response, out.answer)
            
            # Handle navigation state transitions
            self.handle_navigation_response(out)

    def generate_structured_response(self, user_text: str) -> AssistantOutput:
        # Two system messages: static prefix (auto-cached by OpenAI after first call) +
        # small dynamic state block (changes each turn, not cached).
        input_items = [
            {"role": "system", "content": self._static_system_msg},
            {"role": "system", "content": self._build_dynamic_state_msg()},
        ]
        input_items.extend(self.history[-6:])
        input_items.append({"role": "user", "content": user_text})

        # Enable web_search tool; the model chooses whether to call it. :contentReference[oaicite:13]{index=13}
        # Use Structured Outputs for guaranteed schema conformance. :contentReference[oaicite:14]{index=14}
        resp = self.client.responses.parse(
            model=self.response_model,
            tools=[{"type": "web_search"}],
            input=input_items,
            text_format=AssistantOutput,
        )

        parsed: AssistantOutput = resp.output_parsed
        parsed.answer = clamp_text(sanitize_for_tts(parsed.answer), MAX_TTS_CHARS)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": parsed.model_dump_json()})
        # Cap list so memory doesn't grow unboundedly across a long session
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return parsed
    
    def handle_navigation_response(self, out: AssistantOutput) -> None:
        """
        Handle the LLM response and drive FSM state transitions.
        - navigate action: validate location → GUIDING (face_speaker OFF, nav2 owns cmd_vel)
        - speak action while navigating: stay in GUIDING
        - speak action while idle: stay AWAKE so user can continue conversing
        """
        print(f"\n[FSM] Current: {self.robot_state.value}")

        if out.next_action == "navigate":
            location = out.location.value if isinstance(out.location, Enum) else out.location

            if self.state == "guiding" and self._current_navigation_location == location:
                print(f"[NAV] Already guiding to '{location}'. Ignoring duplicate.")
                return

            if location in SPECIAL_LOCATION_TOKENS:
                print(f"[NAV] Cannot navigate to '{location}' — special token, not a destination")
                self._transition_to(RobotState.AWAKE, "invalid location token")
                self._reset_dialog_state()
                return

            loc_info = self.task_info.get("location_info", {})
            if location not in loc_info:
                print(f"[NAV] Location '{location}' not found in location_info")
                self._transition_to(RobotState.AWAKE, "location not found")
                self._reset_dialog_state()
                return

            coords = loc_info[location].get("coordinates")
            if not coords or len(coords) < 2:
                print(f"[NAV] No valid coordinates for '{location}'")
                self._transition_to(RobotState.AWAKE, "no coordinates")
                self._reset_dialog_state()
                return

            x, y = coords[0], coords[1]
            yaw = loc_info[location].get("yaw", 0.0)
            print(f"[NAV] Attempting to navigate to '{location}' at ({x}, {y}) yaw={yaw:.3f} rad")

            path_valid = True
            if self.use_ros and self.ros_node is not None:
                if self.skip_path_check:
                    print("[NAV] Path check skipped (--skip-path-check flag)")
                    path_valid = True
                # else:
                #     print("[NAV] Validating path...")
                #     path_valid = self.ros_node.compute_path_to_pose(x, y)

                if path_valid:
                    # GUIDING: face_speaker disabled, nav2 takes full cmd_vel ownership
                    self._transition_to(RobotState.GUIDING, f"navigating to {location}")
                    self._current_navigation_location = location
                    self.dialog_state = "navigating"
                    self.pending_location = "na"
                    self._set_face("guiding", f"Guiding to {location}")
                    self._refresh_face_gaze()
                    self.ros_node.publish_goal(x, y, location_name=location, yaw=yaw)
                    print(f"[NAV] Navigation goal published for '{location}'")
                else:
                    print(f"[NAV] No valid path found.")
                    self._transition_to(RobotState.AWAKE, "no valid path")
                    self._reset_dialog_state()
            else:
                # ROS2 not available — simulate for testing
                print("[NAV] ROS2 not available — simulating navigation")
                self._transition_to(RobotState.GUIDING, f"simulated navigation to {location}")
                self._current_navigation_location = location
                self.dialog_state = "navigating"
                self.pending_location = "na"
                self._set_face("guiding", f"Guiding to {location}")
                self._refresh_face_gaze()
                print(f"[NAV] Simulated guiding to '{location}'")

        elif out.next_action == "speak":
            if self.state == "guiding":
                # Remain in GUIDING while nav2 is active; just update dialog state
                self.dialog_state = "navigating"
                self.pending_location = "na"
            else:
                # Stay AWAKE so the user can keep conversing without re-triggering wake word
                if self.robot_state not in (RobotState.AWAKE,):
                    self._transition_to(RobotState.AWAKE, "speak response")
                self._set_face("idle", self._idle_face_subtitle())
                self._set_face_gaze(0.0)

        print(f"[FSM] New: {self.robot_state.value}")


# -----------------------------
# 4) Run
# -----------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="GuideBot interaction pipeline"
    )
    parser.add_argument(
        "--text", "-t",
        action="store_true",
        help="Run in text input mode for debugging (instead of streaming audio)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="Response model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--no-ros",
        action="store_true",
        help="Disable ROS2 integration (for debugging without ROS2 environment)"
    )
    parser.add_argument(
        "--skip-path-check",
        action="store_true",
        help="Skip path validation before initiating navigation (navigate directly)"
    )
    parser.add_argument(
    "--audio-backend",
    choices=["auto", "pulse", "alsa"],
    default="auto",
    help="Audio I/O backend. 'pulse' uses parec/pacat, 'alsa' uses sounddevice/PortAudio."
    )
    parser.add_argument("--pulse-source", default=None, help="Pulse source name (from `pactl list short sources`).")
    parser.add_argument("--pulse-sink", default=None, help="Pulse sink name (from `pactl list short sinks`).")
    parser.add_argument("--sd-input-device", default=None, help="sounddevice input device (index or substring).")
    parser.add_argument("--sd-output-device", default=None, help="sounddevice output device (index or substring).")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only process the most recent transcription, discarding stale queued utterances."
    )
    parser.add_argument(
        "--use-wake-word",
        action="store_true",
        help="Enable wake word detection. Robot will only respond after hearing 'hey alpaca'."
    )
    parser.add_argument(
        "--wake-timeout",
        type=float,
        default=60.0,
        help="Seconds to stay 'awake' after hearing wake phrase (default: 30s)."
    )
    parser.add_argument(
        "--wake-phrase",
        type=str,
        default="hey alpaca",
        help="Custom wake phrase (default: 'hey alpaca')."
    )
    parser.add_argument(
        "--face-display",
        action="store_true",
        help="Enable simple expressive face UI on a connected display."
    )
    parser.add_argument(
        "--face-windowed",
        action="store_true",
        help="Run face UI in windowed mode instead of fullscreen."
    )
    parser.add_argument(
        "--dummy-motion",
        action="store_true",
        help="Inject synthetic rollout/cmd_vel turn hints for face gaze testing off-robot."
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.65,
        help=(
            "Server VAD speech-detection threshold (0.0–1.0, default 0.65). "
            "Raise this (e.g. 0.75–0.85) if background noise (lidar, fans) triggers "
            "false speech-started events."
        ),
    )
    parser.add_argument(
        "--noise-filter-hz",
        type=float,
        default=0.0,
        help=(
            "Client-side 4th-order Butterworth high-pass filter cutoff in Hz "
            "(default: disabled). Set to ~150 to cut lidar motor hum and low-freq "
            "mechanical noise before audio is sent to the STT API. Requires scipy."
        ),
    )
    parser.add_argument(
        "--face-speaker",
        action="store_true",
        help=(
            "Enable active-speaker tracking: robot turns to face whoever is speaking "
            "via /talknce/active_speaker_bbox detections. Disabled automatically during "
            "navigation (GUIDING state); re-enabled on arrival or cancel."
        ),
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default="data",
        help="Base directory for transcript run folders (default: ./data).",
    )
    args = parser.parse_args()

    bot = VoiceChatbot(
        system_prompt_path="system_prompt.txt",
        task_info_path="task_info.json",
        transcription_model="gpt-4o-transcribe",
        response_model=args.model,
        text_mode=args.text,
        use_ros=not args.no_ros,
        skip_path_check=args.skip_path_check,
        audio_backend=args.audio_backend,
        pulse_source=args.pulse_source,
        pulse_sink=args.pulse_sink,
        sd_input_device=args.sd_input_device,
        sd_output_device=args.sd_output_device,
        latest_only=args.latest_only,
        use_wake_word=args.use_wake_word,
        wake_timeout=args.wake_timeout,
        wake_phrase=args.wake_phrase,
        use_face_display=args.face_display,
        face_fullscreen=not args.face_windowed,
        dummy_motion=args.dummy_motion,
        transcript_dir=args.transcript_dir,
        use_face_speaker=args.face_speaker,
        vad_threshold=args.vad_threshold,
        noise_filter_hz=args.noise_filter_hz,
    )

    try:
        if args.text:
            await bot.start_text_mode()
        else:
            await bot.start()
    finally:
        bot.shutdown()


def run():
    """Sync entry point for console_scripts (main() is a coroutine; the
    setuptools wrapper calls this directly rather than through asyncio.run)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    run()