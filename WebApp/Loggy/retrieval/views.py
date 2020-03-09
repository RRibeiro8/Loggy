from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import TopicObject
from .forms import ObjectsForm


class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, *args, **kwargs):

		form = ObjectsForm(self.request.POST)
		print(form)

		if form.is_valid():
			
			form = form.save()

			return JsonResponse({"success": True,
			"form": form }, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, *args, **kwargs):

		queryset = TopicObject.objects.all()

		form = ObjectsForm()

		context = { 'queryset': queryset,
					'form': form }
		return render(self.request, self.template_name, context)
