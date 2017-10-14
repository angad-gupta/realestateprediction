from django.db import models

# Create your models here.

class Housing(models.Model):
    location = models.CharField(max_length=250)
    area = models.FloatField()
    price = models.FloatField()

    def __str__(self):
        return self.location


class Facility(models.Model):
    housing = models.ForeignKey(Housing, on_delete=models.CASCADE)
    location = models.CharField(max_length=250)
    main_road = models.BooleanField
    branch_road = models.BooleanField
    inner_road = models.BooleanField

    def __str__(self):
        return self.location


class Land(models.Model):
    date = models.CharField(max_length=50)
    address = models.CharField(max_length=50)
    type_of_land = models.CharField(max_length=50)
    type_of_path = models.CharField(max_length=50)
    facility = models.CharField(max_length=50)
    high_tension = models.CharField(max_length=50)
    river = models.CharField(max_length=50)
    shape = models.CharField(max_length=50)
    level = models.CharField(max_length=50)
    distance = models.FloatField()
    road_width = models.FloatField()
    gov_rate = models.FloatField()
    com_rate = models.FloatField()

    def __str__(self):
        return self.date

