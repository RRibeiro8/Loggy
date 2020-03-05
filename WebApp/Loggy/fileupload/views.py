from django.http import HttpResponse
from django.views.generic import CreateView, DeleteView, ListView
from .models import ImageModel
from .response import JSONResponse, response_mimetype
from .serialize import serialize

import json
import os
from django.conf import settings

class ImageCreateView(CreateView):
    model = ImageModel
    fields = "__all__"

    def form_valid(self, form):
        self.object = form.save(commit=False)
        image_data = self.request.FILES['file'].read()
        image_path = self.request.FILES['file']

        self.object.slug = self.object.file.name
        
        with open(os.path.join(settings.MEDIA_ROOT, 'data.json'), 'r') as f:
            images_info = json.load(f)
                
        form.save()
    
        files = [serialize(self.object)]
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
