from django.shortcuts import render
from django.views import View 

class HomeView(View):

	template_name = 'base.html'

	def get(self, request):
	
		context = {}
		return render(request, self.template_name, context)