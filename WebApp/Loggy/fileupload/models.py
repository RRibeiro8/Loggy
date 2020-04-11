from django.db import models
from django.contrib import admin

class ConceptModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)    

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['-tag']


class ImageModel(models.Model):

    file = models.ImageField(upload_to="database/")
    slug = models.SlugField(max_length=50, blank=True)
    minute_id = models.CharField(max_length=255, blank=True)
    date_time = models.DateTimeField(blank=True, null=True)
    concepts = models.ManyToManyField(ConceptModel, blank=True, through='ConceptScoreModel')#, related_name = 'Concept', through='ConceptScoreModel')

    def __str__(self):
        return self.slug

    class Meta:
        ordering = ['date_time',]

    def get_absolute_url(self):
        return ('upload-new', )

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(ImageModel, self).delete(*args, **kwargs)

class ConceptScoreModel(models.Model):
    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE, null=True)
    concept = models.ForeignKey(ConceptModel, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(null=True)
    box = models.CharField(max_length=255, blank=True)

class ConceptScoreModel_inline(admin.TabularInline):
    model = ConceptScoreModel

class ConceptModelAdmin(admin.ModelAdmin):
    inlines = (ConceptScoreModel_inline,)

class ImageModelAdmin(admin.ModelAdmin):
    inlines = (ConceptScoreModel_inline,)


class LocationModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    latitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    longitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    tag = models.CharField(max_length=255, blank=True, null=True)
    timezone = models.CharField(max_length=255, blank=True)
    local_time = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['local_time',]


