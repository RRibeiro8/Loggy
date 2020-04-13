from django.db import models

class TopicModel(models.Model):

	title = models.CharField(max_length = 255, unique = True)
	description = models.TextField(blank=True, null=True)
	narrative = models.TextField(blank=True, null=True)
	topic_id = models.CharField(max_length = 5, unique = True)

	def __str__(self):
		return self.title

	class Meta:
		ordering = ['title',]

class SimilarityModel(models.Model):
	word1 = models.CharField(max_length = 255)
	word2 = models.CharField(max_length = 255)
	score = models.FloatField()

	def __str__(self):
		return (self.word1 + " - " + self.word2)

	class Meta:
		ordering = ['-score',]

# class TopicObject(models.Model):
# 	tag = models.CharField(max_length = 255, unique = True)

# 	def __str__(self):
# 		return self.tag

# 	class Meta:
# 		ordering = ['tag',]

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

