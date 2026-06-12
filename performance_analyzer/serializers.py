from rest_framework import serializers
from .models import PerformanceVideo

class PerformanceVideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerformanceVideo
        fields = [
            'id',
            'filename',
            'tier',
            'actual_ctr',
            'impressions',
            'brain_predicted_ctr',
            'brain_timeseries'
        ]
