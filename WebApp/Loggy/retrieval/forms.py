from django import forms
from .models import TopicObject

class ObjectsForm(forms.ModelForm):

	class Meta:
		model = TopicObject
		fields = ['tag',]
