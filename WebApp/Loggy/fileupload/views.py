from django.http import HttpResponse
from django.views.generic import CreateView, DeleteView, ListView
from .models import (ImageModel, LocationModel) 
from visualrecognition.models import (ActivityModel, AttributesModel, CategoryModel, ConceptModel)
from .response import JSONResponse, response_mimetype
from .serialize import serialize

import json
import os
from django.conf import settings
from datetime import datetime
import pytz

class ImageCreateView(CreateView):
    model = ImageModel
    fields = "__all__"

    images_info = None

    with open(os.path.join(settings.MEDIA_ROOT, 'data.json'), 'r') as f:
        images_info = json.load(f)

    def form_valid(self, form):
        image = form.save(commit=False)
        image_data = self.request.FILES['file'].read()
        image_path = self.request.FILES['file']

        image.slug = image.file.name

        img_data = self.images_info[image.file.name]
        image.minute_id = img_data['minute_id']

        utc_time = img_data['utc_time']
        dt = datetime.strptime(utc_time, 'UTC_%Y-%m-%d_%H:%M')
        image.date_time = dt.replace(tzinfo=pytz.UTC)
                
        form.save()

        lt = datetime.strptime(img_data['local_time'], '%Y-%m-%d_%H:%M')
        local_time = lt.replace(tzinfo=pytz.timezone(img_data['timezone']))
        LocationModel.objects.create(image=image, latitude=img_data['latitude'], longitude=img_data['longitude'], 
                                    name=img_data['location'], timezone=img_data['timezone'], local_time=local_time)

        ActivityModel.objects.create(image=image, tag=img_data['activity'])

        for attr in img_data['atributtes']:
            AttributesModel.objects.create(image=image, tag=attr)

        for cat in img_data['categories']:
            CategoryModel.objects.create(image=image, tag=cat, score=img_data['categories'][cat])

        for con in img_data['concepts']:
            p_string = ""
            for p in img_data['concepts'][con]['box']:
                p_string = p_string + str(p) + " " 
            ConceptModel.objects.create(image=image, tag=con, score=img_data['concepts'][con]['score'], box=p_string)
    
        files = [serialize(image)]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response

    def form_invalid(self, form):
        data = json.dumps(form.errors)
        return HttpResponse(content=data, status=400, content_type='application/json')

class ImageDeleteView(DeleteView):
    model = ImageModel

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()
        self.object.delete()
        response = JSONResponse(True, mimetype=response_mimetype(request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response


class ImageListView(ListView):
    model = ImageModel

    def render_to_response(self, context, **response_kwargs):
        files = [ serialize(p) for p in self.get_queryset() ]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response
