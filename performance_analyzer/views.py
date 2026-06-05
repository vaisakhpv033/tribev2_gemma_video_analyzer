from rest_framework import generics
from .models import PerformanceVideo
from .serializers import PerformanceVideoSerializer

class PerformanceVideoListView(generics.ListAPIView):
    """
    Returns a list of all PerformanceVideo objects.
    Sorted by actual_ctr in descending order.
    """
    queryset = PerformanceVideo.objects.order_by('-actual_ctr')
    serializer_class = PerformanceVideoSerializer
    pagination_class = None  # We want to return all of them at once for the dropdowns
