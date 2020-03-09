from django.db import models

class TopicObject(models.Model):
	tag = models.CharField(max_length = 255, unique = True)

	def __str__(self):
		return self.tag

	class Meta:
		ordering = ['tag',]

# class TopicLocation(models.Model):
# 	tag = models.CharField(max_length = 255, unique = True)

# 	def __str__(self):
# 		return self.tag

# 	class Meta:
# 		ordering = ['tag',]

# class TopicActivity(models.Model):
# 	tag = models.CharField(max_length = 255, unique = True)
	
# 	def __str__(self):
# 		return self.tag

# 	class Meta:
# 		ordering = ['tag',]

# class TopicOther(models.Model):
# 	tag = models.CharField(max_length = 255, unique = True)

# 	def __str__(self):
# 		return self.tag

# 	class Meta:
# 		ordering = ['tag',]

