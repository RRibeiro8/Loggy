from django.shortcuts import render
from django.views import View
from fileupload.models import ImageModel

class GalleryView(View):

	template_name = "gallery/gallery.html"
	model = ImageModel

	def get(self, request):
		queryset = self.model.objects.all()
		context = { 'queryset': queryset }
		return render(request, self.template_name, context)
