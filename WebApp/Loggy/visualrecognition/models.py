from django.db import models

from fileupload.models import ImageModel

class ActivityModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['image']

class AttributesModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['image']

class CategoryModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    tag = models.CharField(max_length=255, blank=True)
    score = models.FloatField(null=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['-score']

class ConceptModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    tag = models.CharField(max_length=255, blank=True)
    score = models.FloatField(null=True)
    box = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['-score']
