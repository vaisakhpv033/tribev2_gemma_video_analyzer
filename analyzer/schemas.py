from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# Pydantic Schemas for LLMs Structured JSON Output
# =====================================================================

class VideoSegment(BaseModel):
    timestamp_start: str = Field(description="Start time of the segment/shot (e.g., '00:00')")
    timestamp_end: str = Field(description="End time of the segment/shot (e.g., '00:03')")
    visuals: str = Field(description="Visual description: characters, environment clutter, gameplay actions, text overlays, hand/pointer guides, UI elements (timers, menus)")
    audio: str = Field(description="Audio description: voiceover, character sounds/cries, game sound effects (clicks, success/fail chimes), music tempo/mood")
    pacing_and_emotion: str = Field(description="Pacing level (Fast/Medium/Slow) and core emotions triggered (e.g., urgency, stress, satisfaction, curiosity)")


class GameAdTropeAnalysis(BaseModel):
    ad_format_type: str = Field(description="Type of mobile game ad format (e.g., 'Fail Gameplay', 'Success/Satisfying Gameplay', 'Narrative Drama/Story-focused', 'Interactive/Choice Renovation', 'Mixed')")
    has_story_narrative: bool = Field(description="Whether the ad showcases a story or narrative sequence (e.g., character in distress, renovation theme, betrayal story)")
    story_summary: Optional[str] = Field(description="Brief summary of the narrative plot shown in the ad (e.g., helping a cold/dirty character, restoring a room, solving a family conflict)")
    gameplay_type_shown: str = Field(description="Specific gaming mechanics shown (e.g., 'Hidden Object Search', 'Decorating/Renovation Choice', 'Match-3', 'Not showing actual gameplay')")
    is_fail_ad: bool = Field(description="Does the virtual player deliberately make a wrong choice, fail the search, or fail the level to frustrate/engage the viewer?")


class HookAnalysis(BaseModel):
    scroll_stopper_rating: int = Field(description="Rating from 1-10 on how effective the first 3 seconds are at capturing casual gamers' attention")
    hook_type: str = Field(description="Type of hook used (e.g., 'Extreme Clutter/Mess', 'Emotional Character Drama', 'Urgent Countdown', 'Direct Puzzle Gameplay')")
    analysis: str = Field(description="Detailed analysis of why the hook works or fails for the target audience")
    suggestions: str = Field(description="Actionable improvement to make the initial hook even more scroll-stopping")


class MobileGameAdCreativeAudit(BaseModel):
    what_went_well: List[str] = Field(description="Strengths of the ad (e.g., highly satisfying cleaning sequence, clear hidden objects, engaging hand pointer, strong emotional hook, early branding)")
    what_went_wrong: List[str] = Field(description="Weaknesses/Friction points (e.g., hidden objects are too tiny/unclear, fake gameplay is too obvious, CTA is generic, story lacks resolution, pacing is too slow)")
    actionable_feedback: List[str] = Field(description="Step-by-step optimization recommendations for mobile game performance (e.g., testing a 'satisfying clean' vs. 'dramatic fail' version)")


class AdVideoAnalysis(BaseModel):
    timeline: List[VideoSegment] = Field(description="Detailed chronological breakdown of the ad segment-by-segment")
    trope_analysis: GameAdTropeAnalysis = Field(description="Analysis of mobile game advertising hooks and tropes represented in the video")
    hook: HookAnalysis = Field(description="Evaluation of the first 3 seconds of the ad")
    audit: MobileGameAdCreativeAudit = Field(description="Expert mobile gaming ad creative audit with strengths, weaknesses, and recommendations")
    creative_score: float = Field(description="Overall creative performance score from 1.0 to 10.0 based on its potential to drive App Store installs/CTR")
