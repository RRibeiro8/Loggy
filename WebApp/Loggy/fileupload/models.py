from django.db import models

class ImageModel(models.Model):

    file = models.ImageField(upload_to="database/")
    slug = models.SlugField(max_length=50, blank=True)
    minute_id = models.CharField(max_length=255, blank=True)
    date_time = models.DateTimeField(blank=True, null=True)

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


class LocationModel(models.Model):

    image = models.ForeignKey(ImageModel, on_delete=models.CASCADE)
    latitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    longitude = models.DecimalField(blank=True, null=True, max_digits=11, decimal_places=8)
    name = models.CharField(max_length=255, blank=True)
    timezone = models.CharField(max_length=255, blank=True)
    local_time = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.image.slug

    class Meta:
        ordering = ['local_time',]

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
