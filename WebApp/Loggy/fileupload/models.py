from django.db import models
from django.contrib import admin

class ConceptModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)    

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['tag']

class LocationModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['tag']

class CategoryModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['tag']

class ActivityModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['tag']

class AttributesModel(models.Model):

    tag = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.tag

    class Meta:
        ordering = ['tag']


class ImageModel(models.Model):

    file = models.ImageField(upload_to="database/")
    slug = models.SlugField(max_length=50, blank=True)
    minute_id = models.CharField(max_length=255, blank=True)
    date_time = models.DateTimeField(blank=True, null=True)
    concepts = models.ManyToManyField(ConceptModel, blank=True, through='ConceptScoreModel')
    location = models.ManyToManyField(LocationModel, blank=True, through='LocationInfoModel')
    categories = models.ManyToManyField(CategoryModel, blank=True, through='CategoryScoreModel')
    activities = models.ManyToManyField(ActivityModel, blank=True, through='ActivityInfoModel')
    attributes = models.ManyToManyField(AttributesModel, blank=True, through='AttributesInfoModel')

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
    tag = models.ForeignKey(ConceptModel, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(null=True)
    box = models.CharField(max_length=255, blank=True)

class LocationInfoModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    tag = models.ForeignKey(LocationModel, on_delete=models.CASCADE)
    latitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    longitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    timezone = models.CharField(max_length=255, blank=True)
    local_time = models.DateTimeField(blank=True, null=True)

class CategoryScoreModel(models.Model):
    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE, null=True)
    tag = models.ForeignKey(CategoryModel, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField(null=True)

class ActivityInfoModel(models.Model):
    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE, null=True)
    tag = models.ForeignKey(ActivityModel, on_delete=models.CASCADE, null=True, blank=True)

class AttributesInfoModel(models.Model):
    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE, null=True)
    tag = models.ForeignKey(AttributesModel, on_delete=models.CASCADE, null=True, blank=True)

class ConceptScoreModel_inline(admin.TabularInline):
    model = ConceptScoreModel
    extra = 1

class LocationInfoModel_inline(admin.TabularInline):
    model = LocationInfoModel
    extra = 1

class CategoryScoreModel_inline(admin.TabularInline):
    model = CategoryScoreModel
    extra = 1

class ActivityInfoModel_inline(admin.TabularInline):
    model = ActivityInfoModel
    extra = 1

class AttributesInfoModel_inline(admin.TabularInline):
    model = AttributesInfoModel
    extra = 1

class ConceptModelAdmin(admin.ModelAdmin):
    inlines = (ConceptScoreModel_inline,)

class LocationModelAdmin(admin.ModelAdmin):
    inlines = (LocationInfoModel_inline,)

class CategoryModelAdmin(admin.ModelAdmin):
    inlines = (CategoryScoreModel_inline,)

class ActivityModelAdmin(admin.ModelAdmin):
    inlines = (ActivityInfoModel_inline,)

class AttributesModelAdmin(admin.ModelAdmin):
    inlines = (AttributesInfoModel_inline,)

class ImageModelAdmin(admin.ModelAdmin):
    inlines = (ConceptScoreModel_inline, LocationInfoModel_inline, CategoryScoreModel_inline, 
        ActivityInfoModel_inline, AttributesInfoModel_inline)




