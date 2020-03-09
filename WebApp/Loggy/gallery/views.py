from django.shortcuts import render
from django.views import View
from fileupload.models import ImageModel

from collections import Counter

from datetime import datetime

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

		obj_list = [obj.date_time.date() for obj in self.model.objects.all()]
		counter = Counter(obj_list)
		l = [obj for obj in counter]
		l.sort()

		queryset = {}

		for date in l:

			obj = self.model.objects.filter(date_time__contains=date)#date_time__year=year,)# date_time__month=month, date_time__day=day)
			#print(obj)
			tmp = date.strftime('%Y-%m-%d')
			queryset[tmp] = obj

		#print(queryset)

		context = { 'queryset': queryset }
		return render(request, self.template_name, context)