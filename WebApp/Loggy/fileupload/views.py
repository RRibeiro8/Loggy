from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import CreateView, DeleteView, ListView
from django.views import View
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
from tqdm import tqdm

from retrieval.sentence_analyzer.nlp_analyzer import word2lemma

class UpdateLocationsView(View):

    template_name = "fileupload/update_locations.html"
    model = LocationModel

    def get(self, request, *args, **kwargs):

        locations = self.model.objects.all()
        context = { 'locations': locations }
        return render(self.request, self.template_name, context)

    def post(self, request, *args, **kwargs):

        method = request.POST.getlist('button')[0]

        image_query = ImageModel.objects.all()
        if method == "update":
            print("Updating locations")

                                     
        if method == "delete":
            print("deleting locations")

        locations = self.model.objects.all()
        context = { 'locations': locations }
        return render(self.request, self.template_name, context)

class UpdateActivitiesView(View):

    template_name = "fileupload/update_activities.html"
    model = ActivityModel

    def get(self, request, *args, **kwargs):

        activities = self.model.objects.all()
        context = { 'activities': activities }
        return render(self.request, self.template_name, context)

    def post(self, request, *args, **kwargs):

        method = request.POST.getlist('button')[0]

        image_query = ImageModel.objects.all()
        if method == "update":
            print("Updating Activities")

                                     
        if method == "delete":
            print("deleting Activities")

        activities = self.model.objects.all()
        context = { 'activities': activities }
        return render(self.request, self.template_name, context)

class UpdateConceptsView(View):

    template_name = "fileupload/update_concepts.html"
    model = ConceptModel

    def get(self, request, *args, **kwargs):

        concepts = self.model.objects.all()
        context = { 'concepts': concepts }
        return render(self.request, self.template_name, context)

    def post(self, request, *args, **kwargs):

        method = request.POST.getlist('button')[0]

        image_query = ImageModel.objects.all()
        if method == "update":
            print("Updating Concepts")

            with open(os.path.join(settings.MEDIA_ROOT, 'updated_concepts.json'), 'r') as concepts_file:
                images_concepts_info = json.load(concepts_file)

                for img in tqdm(image_query):
                    
                    new_concepts = images_concepts_info[img.slug]
                    #print(img.concepts.all())
                    img.concepts.clear()
                    #print(img.concepts.all())

                    for con in new_concepts["concepts"]:
                        if con:
                            for c in con:

                                #print(con[c])
                                p_string = ""
                                for p in con[c]['box']:
                                    p_string = p_string + str(p) + " " 

                                con_obj = word2lemma(c)
                                if (ConceptModel.objects.filter(tag=con_obj)):
                                    obj = ConceptModel.objects.get(tag=con_obj)
                                else: 
                                    print(con_obj, " doesn't exist...Creating")
                                    obj = ConceptModel(tag=con_obj)
                                    obj.save()

                                ConceptScoreModel.objects.create(image=img, tag=obj, score=con[c]['score'], box=p_string)

                        else: 
                            print("No Concepts!")

                    #print(img.concepts.all())                                    


        if method == "delete":
            print("deleting Concepts")

        concepts = self.model.objects.all()
        context = { 'concepts': concepts }
        return render(self.request, self.template_name, context)


class ImageCreateView(CreateView):
    model = ImageModel
    fields = "__all__"

    images_info = None

    with open(os.path.join(settings.MEDIA_ROOT, 'data.json'), 'r') as f:
        images_info = json.load(f)

    def form_valid(self, form):
        
        #image_data = self.request.FILES['file'].read()
        #image_path = self.request.FILES['file']
        #print(image_data)

        files = []

        try:
            image = form.save(commit=False)
            img_data = self.images_info[image.file.name]

            image.slug = image.file.name

            image.minute_id = img_data['minute_id']

            utc_time = img_data['utc_time']
            dt = datetime.strptime(utc_time, 'UTC_%Y-%m-%d_%H:%M')
            image.date_time = dt.replace(tzinfo=pytz.UTC)
                    
            form.save()

            for con in img_data['concepts']:
                p_string = ""
                for p in img_data['concepts'][con]['box']:
                    p_string = p_string + str(p) + " " 

                con_obj = word2lemma(con)
                if (ConceptModel.objects.filter(tag=con_obj)):
                    obj = ConceptModel.objects.get(tag=con_obj)
                else: 
                    obj = ConceptModel(tag=con_obj)
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
                loc_obj = img_data['location']
                if (LocationModel.objects.filter(tag=loc_obj)):
                    location = LocationModel.objects.get(tag=loc_obj)   
                else:
                    location = LocationModel(tag=loc_obj)
                    location.save()


            LocationInfoModel.objects.create(image=image, tag=location, latitude=img_data['latitude'], longitude=img_data['longitude'], 
                                                timezone=img_data['timezone'], local_time=local_time)

            for cat in img_data['categories']:
                tmp = cat.split('/')
                cat_filtered = tmp[0].replace('_', ' ')

                cat_obj = word2lemma(cat_filtered)
                if (CategoryModel.objects.filter(tag=cat_obj)):
                    category = CategoryModel.objects.get(tag=cat_obj)
                else:
                    category = CategoryModel(tag=cat_obj)
                    category.save()
                
                CategoryScoreModel.objects.create(image=image, tag=category, score=img_data['categories'][cat])

            activity = None
            if img_data['activity'] != "NULL":
                activity = img_data['activity'] #one_word2lemma(img_data['activity'])
                act_obj = word2lemma(activity)
                if (ActivityModel.objects.filter(tag=act_obj)):
                    activity = ActivityModel.objects.get(tag=act_obj)
                else:
                    activity = ActivityModel(tag=act_obj)
                    activity.save()

            ActivityInfoModel.objects.create(image=image, tag=activity)

            attribute = None
            for attr in img_data['atributtes']:
                lemma_attr = word2lemma(attr)#one_word2lemma(attr)

                if (AttributesModel.objects.filter(tag=lemma_attr)):
                    attribute = AttributesModel.objects.get(tag=lemma_attr)
                else:
                    attribute = AttributesModel(tag=lemma_attr)
                    attribute.save()

            AttributesInfoModel.objects.create(image=image, tag=attribute)

            files = [serialize(image)]
        except:
            print("Image - NO DATA! - Error")
            #form.save()
            
        #files = [serialize(image)]
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
