from typing import List
from pydantic import BaseModel, Field

class VideoStrengthsWeaknesses(BaseModel):
    what_went_well: List[str] = Field(description="Strengths of this specific video (e.g., highly satisfying sequence, clear messaging, engaging hook)")
    what_went_wrong: List[str] = Field(description="Weaknesses of this specific video (e.g., confusing UI, pacing too slow, poor audio sync)")
    gameplay_and_narrative_clarity: str = Field(description="Evaluation of how clearly the core loop is shown, whether it's an identifiable hidden object puzzle, a renovation choice, or clear story progression.")
    neural_alignment: str = Field(description="How well does the creative execution align with the neural cognitive scores? Do the strengths/weaknesses map correctly to the brain data?")

class HookComparison(BaseModel):
    video1_hook_rating: int = Field(description="Rating from 1-10 for Video 1's hook")
    video2_hook_rating: int = Field(description="Rating from 1-10 for Video 2's hook")
    attention_capture_analysis: str = Field(description="Detailed comparison of the first 3 seconds of both videos, explicitly correlating the visual/audio hooks with the 'Attention Capture' and 'Novelty Salience' neural scores.")
    better_hook_video: str = Field(description="Which video had a better hook: 'video1', 'video2', or 'tie'")

class MessagingComparison(BaseModel):
    video1_core_message: str = Field(description="The core value proposition or CTA delivered in Video 1")
    video2_core_message: str = Field(description="The core value proposition or CTA delivered in Video 2")
    cognitive_retention_analysis: str = Field(description="Detailed comparison of messaging clarity and CTA effectiveness, explicitly correlating them with the 'Memory Encoding' and 'Narrative Language' neural scores.")

class ComparativeAudit(BaseModel):
    winner: str = Field(description="The overall better performing video for UA: 'video1', 'video2', or 'tie'")
    neural_backed_winner_justification: str = Field(description="Detailed explanation of why the winning video is more likely to perform better (CTR, Conversion), backed by BOTH creative tropes and the quantitative neural data.")
    actionable_hybrid_idea: str = Field(description="How could you combine the best elements of both videos into a new, superior 'Video 3'?")

class AdVideoComparisonResult(BaseModel):
    video1_analysis: VideoStrengthsWeaknesses = Field(description="Strengths and weaknesses specific to Video 1")
    video2_analysis: VideoStrengthsWeaknesses = Field(description="Strengths and weaknesses specific to Video 2")
    hook_comparison: HookComparison = Field(description="Comparison of the first 3 seconds (hooks) of both videos")
    messaging_comparison: MessagingComparison = Field(description="Comparison of the core message and CTA")
    audit: ComparativeAudit = Field(description="Final comparative audit and winner selection")
