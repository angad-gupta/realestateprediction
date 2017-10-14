<script>
var chart = AmCharts.makeChart("chartdivBar", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "district": "Kathmandu",
    "data": 693,
    "color": "#FF0F00"
  }, {
    "district": "Lalitpur",
    "data": 59,
    "color": "#FF6600"
  }, {
    "district": "Bhaktapur",
    "data": 108,
    "color": "#FF9E01"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "district",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});


var chart = AmCharts.makeChart( "chartdivPie", {
  "type": "pie",
  "theme": "light",
  "dataProvider": [ {
    "country": "Kathmandu",
    "litres": 266.82
  }, {
    "country": "Lalitpur",
    "litres": 22.71
  }, {
    "country": "Bhaktapur",
    "litres": 40.29
  }],
  "valueField": "litres",
  "titleField": "country",
   "balloon":{
   "fixedPosition":true
  },
  "export": {
    "enabled": true
  }
} );


var chart = AmCharts.makeChart("LandKathmandu", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "type_of_land": "Agriculture",
    "data": 22,
    "color": "#8a45ff"
  }, {
    "type_of_land": "Commercial",
    "data": 106,
    "color": "#ff359f"
  }, {
    "type_of_land": "Residental",
    "data": 807,
    "color": "#12daa7"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_land",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});
var chart = AmCharts.makeChart("LandLalitpur", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "type_of_land": "Agriculture",
    "data": 3,
    "color": "#8a45ff"
  }, {
    "type_of_land": "Commercial",
    "data": 7,
    "color": "#ff359f"
  }, {
    "type_of_land": "Residental",
    "data": 48,
    "color": "#12daa7"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_land",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});
var chart = AmCharts.makeChart("LandBhaktapur", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "type_of_land": "Agriculture",
    "data": 9,
    "color": "#8a45ff"
  }, {
    "type_of_land": "Commercial",
    "data": 6,
    "color": "#ff359f"
  }, {
    "type_of_land": "Residental",
    "data": 88,
    "color": "#12daa7"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_land",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});


var chart = AmCharts.makeChart("PathKathmandu", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "type_of_path": "Earthen",
    "data": 308,
    "color": "#3591ff"
  }, {
    "type_of_path": "Goreto",
    "data": 176,
    "color": "#ff362c"
  }, {
    "type_of_path": "Gravelled",
    "data": 35,
    "color": "#FF9E01"
  }, {
    "type_of_path": "Pitched",
    "data": 161,
    "color": "#e1ff19"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_path",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});

var chart = AmCharts.makeChart("PathLalitpur", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider": [{
    "type_of_path": "Earthen",
    "data": 26,
    "color": "#3591ff"
  }, {
    "type_of_path": "Goreto",
    "data": 19,
    "color": "#ff362c"
  }, {
    "type_of_path": "Gravelled",
    "data": 3,
    "color": "#FF9E01"
  }, {
    "type_of_path": "Pitched",
    "data": 3,
    "color": "#e1ff19"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_path",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});

var chart = AmCharts.makeChart("PathBhaktapur", {
  "type": "serial",
  "theme": "light",
  "marginRight": 70,
  "dataProvider":[{
    "type_of_path": "Earthen",
    "data": 47,
    "color": "#3591ff"
  }, {
    "type_of_path": "Goreto",
    "data": 36,
    "color": "#ff362c"
  }, {
    "type_of_path": "Gravelled",
    "data": 3,
    "color": "#FF9E01"
  }, {
    "type_of_path": "Pitched",
    "data": 17,
    "color": "#e1ff19"
  }],
  "valueAxes": [{
    "axisAlpha": 0,
    "position": "left",
    "title": "Data"
  }],
  "startDuration": 1,
  "graphs": [{
    "balloonText": "<b>[[category]]: [[value]]</b>",
    "fillColorsField": "color",
    "fillAlphas": 0.9,
    "lineAlpha": 0.2,
    "type": "column",
    "valueField": "data"
  }],
  "chartCursor": {
    "categoryBalloonEnabled": false,
    "cursorAlpha": 0,
    "zoomable": false
  },
  "categoryField": "type_of_path",
  "categoryAxis": {
    "gridPosition": "start",
    "labelRotation": 45
  },
  "export": {
    "enabled": true
  }

});


var chart = AmCharts.makeChart("chartdiv", {
  "type": "serial",
     "theme": "light",
  "categoryField": "year",
  "rotate": true,
  "startDuration": 1,
  "categoryAxis": {
    "gridPosition": "start",
    "position": "left"
  },
  "trendLines": [],
  "graphs": [
    {
      "balloonText": "Income:[[value]]",
      "fillAlphas": 0.8,
      "id": "AmGraph-1",
      "lineAlpha": 0.2,
      "title": "Income",
      "type": "column",
      "valueField": "income"
    },
    {
      "balloonText": "Expenses:[[value]]",
      "fillAlphas": 0.8,
      "id": "AmGraph-2",
      "lineAlpha": 0.2,
      "title": "Expenses",
      "type": "column",
      "valueField": "expenses"
    },
        {
      "balloonText": "Expenses:[[value]]",
      "fillAlphas": 0.8,
      "id": "AmGraph-2",
      "lineAlpha": 0.2,
      "title": "third",
      "type": "column",
      "valueField": "third"
    }
  ],
  "guides": [],
  "valueAxes": [
    {
      "id": "ValueAxis-1",
      "position": "top",
      "axisAlpha": 0
    }
  ],
  "allLabels": [],
  "balloon": {},
  "titles": [],
  "dataProvider": [
    {
      "year": 2005,
      "income": 23.5,
      "expenses": 18.1,
            "third":322.1,
            "color": "#ff2f28"
    },
    {
      "year": 2006,
      "income": 26.2,
      "expenses": 22.8
    },
    {
      "year": 2007,
      "income": 30.1,
      "expenses": 23.9
    },
    {
      "year": 2008,
      "income": 29.5,
      "expenses": 25.1
    },
    {
      "year": 2009,
      "income": 24.6,
      "expenses": 25
    }
  ],
    "export": {
      "enabled": true
     }

});

</script>
