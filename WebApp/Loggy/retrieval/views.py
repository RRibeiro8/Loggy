from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import TopicObject
from .forms import ObjectsForm


class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			print(request.POST.getlist('obj_tags[]'))

			return JsonResponse({"success": True}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)
