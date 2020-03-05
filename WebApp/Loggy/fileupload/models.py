from django.db import models

class ImageModel(models.Model):

    image = models.ImageField(upload_to="database/")
    slug = models.SlugField(max_length=50, blank=True)

    def __str__(self):
        return self.slug

    class Meta:
        ordering = ['slug',]

    def get_absolute_url(self):
        return ('upload', )

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """delete -- Remove to leave file."""
        self.file.delete(False)
        super(ImageModel, self).delete(*args, **kwargs)