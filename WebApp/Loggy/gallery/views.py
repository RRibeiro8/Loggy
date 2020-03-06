from django.shortcuts import render
from django.views import View
from fileupload.models import ImageModel

from collections import Counter

class GalleryView(View):

	template_name = "gallery/gallery.html"
	model = ImageModel

	def get(self, request):
		queryset = self.model.objects.all()
		context = { 'queryset': queryset }
		return render(request, self.template_name, context)

class DateTimeView(View):

	template_name = "gallery/datetime.html"
	model = ImageModel

	def get(self, request):

		obj_list = [obj.date_time.strftime('%Y-%m-%d') for obj in self.model.objects.all()]
		counter = Counter(obj_list)
		l = [obj for obj in counter]
		l.sort()

		for i in l:
			
			obj = self.model.objects.filter(date_time=)
			queryset[i] = obj[0].file.url

		queryset = self.model.objects.all()

		context = { 'queryset': queryset }
		return render(request, self.template_name, context)