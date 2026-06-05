from django.contrib import admin
from .models import PerformanceVideo

@admin.register(PerformanceVideo)
class PerformanceVideoAdmin(admin.ModelAdmin):
    list_display = (
        "filename",
        "tier",
        "actual_ctr",
        "brain_predicted_ctr",
        "brain_predicted_class",
        "created_at",
    )
    list_filter = ("tier", "brain_predicted_class")
    search_fields = ("filename",)
    readonly_fields = (
        "brain_predicted_ctr",
        "brain_predicted_class",
        "brain_predicted_confidence",
        "brain_prediction_tier",
        "brain_ctr_lower_bound",
        "brain_ctr_upper_bound",
        "brain_model_features",
        "brain_timeseries",
        "created_at",
        "updated_at",
    )
    
    fieldsets = (
        ("Basic Info", {
            "fields": ("filename", "tier", "npz_file")
        }),
        ("Performance Metrics", {
            "fields": ("actual_ctr",)
        }),
        ("Brain Predictions", {
            "fields": (
                "brain_predicted_ctr",
                "brain_predicted_class",
                "brain_predicted_confidence",
                "brain_prediction_tier",
                "brain_ctr_lower_bound",
                "brain_ctr_upper_bound",
            ),
        }),
        ("Brain Details", {
            "fields": ("brain_model_features", "brain_timeseries"),
            "classes": ("collapse",),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",),
        }),
    )
