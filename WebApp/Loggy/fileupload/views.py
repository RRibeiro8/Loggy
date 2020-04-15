from django.http import HttpResponse
from django.views.generic import CreateView, DeleteView, ListView
from .models import (ImageModel, LocationModel, ConceptModel, ConceptScoreModel, 
                        LocationInfoModel, CategoryModel, CategoryScoreModel,
                        ActivityModel, ActivityInfoModel, AttributesModel, AttributesInfoModel) 
from .response import JSONResponse, response_mimetype
from .serialize import serialize

import json
import os
from django.conf import settings
from datetime import datetime
import pytz

from retrieval.sentence_analyzer.nlp_analyzer import one_word2lemma

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

        try:
            img_data = self.images_info[image.file.name]
            image.minute_id = img_data['minute_id']

            utc_time = img_data['utc_time']
            dt = datetime.strptime(utc_time, 'UTC_%Y-%m-%d_%H:%M')
            image.date_time = dt.replace(tzinfo=pytz.UTC)
                    
            form.save()

            for con in img_data['concepts']:
                p_string = ""
                for p in img_data['concepts'][con]['box']:
                    p_string = p_string + str(p) + " " 

                if (ConceptModel.objects.filter(tag=con)):
                    obj = ConceptModel.objects.get(tag=con)
                else: 
                    obj = ConceptModel(tag=con)
                    obj.save()

                ConceptScoreModel.objects.create(image=image, tag=obj, score=img_data['concepts'][con]['score'], box=p_string)

            lt = datetime.strptime(img_data['local_time'], '%Y-%m-%d_%H:%M')
            local_time = lt.replace(tzinfo=pytz.timezone(img_data['timezone']))
            location = None
            if img_data['location'] == "NULL":
                if (LocationModel.objects.filter(tag='Unknown')):
                    location = LocationModel.objects.get(tag='Unknown')
                else:
                    location = LocationModel(tag='Unknown')
                    location.save()
            else:
                if (LocationModel.objects.filter(tag=img_data['location'])):
                    location = LocationModel.objects.get(tag=img_data['location'])   
                else:
                    location = LocationModel(tag=img_data['location'])
                    location.save()


            LocationInfoModel.objects.create(image=image, tag=location, latitude=img_data['latitude'], longitude=img_data['longitude'], 
                                                timezone=img_data['timezone'], local_time=local_time)

            for cat in img_data['categories']:
                tmp = cat.split('/')
                cat_filtered = tmp[0].replace('_', ' ')

                if (CategoryModel.objects.filter(tag=cat_filtered)):
                    category = CategoryModel.objects.get(tag=cat_filtered)
                else:
                    category = CategoryModel(tag=cat_filtered)
                    category.save()
                
                CategoryScoreModel.objects.create(image=image, tag=category, score=img_data['categories'][cat])

            activity = None
            if img_data['activity'] != "NULL":
                activity = img_data['activity'] #one_word2lemma(img_data['activity'])
                if (ActivityModel.objects.filter(tag=img_data['activity'])):
                    activity = ActivityModel.objects.get(tag=img_data['activity'])
                else:
                    activity = ActivityModel(tag=img_data['activity'])
                    activity.save()

            ActivityInfoModel.objects.create(image=image, tag=activity)

            attribute = None
            for attr in img_data['atributtes']:
                lemma_attr = attr#one_word2lemma(attr)
                if (AttributesModel.objects.filter(tag=lemma_attr)):
                    attribute = AttributesModel.objects.get(tag=lemma_attr)
                else:
                    attribute = AttributesModel(tag=lemma_attr)
                    attribute.save()

            AttributesInfoModel.objects.create(image=image, tag=attribute)

        except:
            form.save()
            
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
        data = {'files': files[0:10]}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response
