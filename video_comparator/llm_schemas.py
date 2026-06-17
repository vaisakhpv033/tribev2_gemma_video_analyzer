from typing import List
from pydantic import BaseModel, Field

class VideoStrengthsWeaknesses(BaseModel):
    what_went_well: List[str] = Field(description="Strengths of this specific video (e.g., highly satisfying sequence, clear messaging, engaging hook)")
    what_went_wrong: List[str] = Field(description="Weaknesses of this specific video (e.g., confusing UI, pacing too slow, poor audio sync)")

class HookComparison(BaseModel):
    video1_hook_rating: int = Field(description="Rating from 1-10 for Video 1's hook")
    video2_hook_rating: int = Field(description="Rating from 1-10 for Video 2's hook")
    comparison_analysis: str = Field(description="Detailed comparison of the first 3 seconds of both videos. Which captures attention better and why?")
    better_hook_video: str = Field(description="Which video had a better hook: 'video1', 'video2', or 'tie'")

class MessagingComparison(BaseModel):
    video1_core_message: str = Field(description="The core value proposition or CTA delivered in Video 1")
    video2_core_message: str = Field(description="The core value proposition or CTA delivered in Video 2")
    comparison_analysis: str = Field(description="Detailed comparison of messaging clarity, CTA effectiveness, and emotional resonance.")

class ComparativeAudit(BaseModel):
    winner: str = Field(description="The overall better performing video for UA: 'video1', 'video2', or 'tie'")
    why_winner_won: str = Field(description="Detailed explanation of why the winning video is more likely to perform better (CTR, Conversion).")
    actionable_hybrid_idea: str = Field(description="How could you combine the best elements of both videos into a new, superior 'Video 3'?")

class AdVideoComparisonResult(BaseModel):
    video1_analysis: VideoStrengthsWeaknesses = Field(description="Strengths and weaknesses specific to Video 1")
    video2_analysis: VideoStrengthsWeaknesses = Field(description="Strengths and weaknesses specific to Video 2")
    hook_comparison: HookComparison = Field(description="Comparison of the first 3 seconds (hooks) of both videos")
    messaging_comparison: MessagingComparison = Field(description="Comparison of the core message and CTA")
    audit: ComparativeAudit = Field(description="Final comparative audit and winner selection")
