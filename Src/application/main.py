from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivy.config import Config
from kivy.core.window import Window
from kivymd.theming import ThemeManager
from kivy.uix.screenmanager import FadeTransition
from kivy.uix.scrollview import ScrollView
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.menu import MDDropdownMenu
from kivy.clock import Clock
import math
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.button import MDFlatButton
from tensorflow import keras
from keras.models import load_model
import lime
import lime.lime_tabular
import numpy as np
import pickle
from kivymd.uix.slider import MDSlider
from PIL import Image
from kivymd.uix.button import MDIconButton
from kivy.uix.popup import Popup



Builder.load_string("""

<FirstScreen>:
    MDLabel:
        font_style: "H4"
        color: 248/255,248/255,255/255,1
        text: "Corona App"
        halign: "center"
        pos_hint: {"center_x": 0.5, "center_y": 0.9}
        size_hint_x: 0.5
        size_hint_y: 0.5
        
    Image:
        source: 'coronavirus.png'
        pos_hint: {"center_x": 0.5, "center_y": 0.55}
        size_hint_x: 0.5
        size_hint_y: 0.5
     
     
    Image:
        source: 'copyright_icon.png'   
        pos_hint: {"center_x": 0.5, "center_y": 0.272}
        size_hint_x: 0.04
        size_hint_y: 0.04
        
    MDLabel:

        font_style: "Overline"
        color: 248/255,248/255,255/255,1
        text: "ofer    montano\\nnoam         hadad\\nFreddy   Gabbay\\nShirly     Bar-Lev"
        halign: "center"
        pos_hint: {"center_x": 0.5, "center_y": 0.22}
        size_hint_x: 0.5
        size_hint_y: 0.5    
        
    MDRaisedButton:
        text: "start"
        md_bg_color: 128/255,128/255,128/255,1
        pos_hint: {"center_x": 0.5, "center_y": 0.1}
        size_hint_x: 0.3
        size_hint_y: 0.065
        on_release: root.manager.current = 'SecondScreen'
        

<SecondScreen>:
    MDLabel:
        font_style: "H5"
        color: 248/255,248/255,255/255,1
        text: "Patient Details"
        halign: "center"
        pos_hint: {"center_x": 0.5, "center_y": 0.9}
        size_hint_x: 0.5
        size_hint_y: 0.5
  
    ScrollView:
        do_scroll_x: False
        do_scroll_y: True
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        size_hint_x: 0.9
        size_hint_y: 0.7
        
        MDBoxLayout:
            id: form
            orientation: "vertical"
            size_hint_y: None
            height: self.minimum_height
                
            MDTextField:
                hint_text: "sex"
                id: sex
                on_focus: if self.focus: {app.sex_menu.open(), self.set_text("",""), app.set_null("sex")}
                input_filter: app.block_typing
                                
            MDTextField:
                hint_text: 'HSD'
                id: HSD
                input_filter: app.unsigned_int
                on_text: app.set_data('HSD',self.text)
                on_focus: if self.focus: {self.set_text("",""), app.set_null("HSD")}

            MDTextField:
                hint_text: "entry month"  
                id: entry_month
                on_focus: if self.focus: {app.entry_month_menu.open(), self.set_text("",""), app.set_null("entry_month")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "symptoms month"  
                id: symptoms_month
                on_focus: if self.focus: {app.symptoms_month_menu.open(), self.set_text("",""), app.set_null("symptoms_month")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "pneumonia" 
                id: pneumonia
                on_focus: if self.focus: {app.pneumonia_menu.open(), self.set_text("",""), app.set_null("pneumonia")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "age" 
                id: age
                input_filter: app.unsigned_int
                on_focus: if self.focus: {self.set_text("",""), app.set_null("age")}
                on_text: app.set_data('age',self.text)

            MDTextField:
                hint_text: "pregnancy" 
                id: pregnancy
                on_focus: if self.focus: {app.pregnancy_menu.open(), self.set_text("",""), app.set_null("pregnancy")}
                input_filter: app.block_typing  
            
            MDTextField:
                hint_text: "diabetes"   
                id: diabetes
                on_focus: if self.focus: {app.diabetes_menu.open(), self.set_text("",""), app.set_null("diabetes")} 
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "copd"     
                id: copd
                on_focus: if self.focus: {app.copd_menu.open(), self.set_text("",""), app.set_null("copd")}
                input_filter: app.block_typing              
            
            MDTextField:
                hint_text: "asthma"   
                id: asthma
                on_focus: if self.focus: {app.asthma_menu.open(), self.set_text("",""), app.set_null("asthma")}  
                input_filter: app.block_typing

            MDTextField:
                hint_text: "inmsupr"  
                id: inmsupr
                on_focus: if self.focus: {app.inmsupr_menu.open(), self.set_text("",""), app.set_null("inmsupr")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "hypertension"  
                id: hypertension
                on_focus: if self.focus: {app.hypertension_menu.open(), self.set_text("",""), app.set_null("hypertension")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "other disease" 
                id: other_disease
                on_focus: if self.focus: {app.other_disease_menu.open(), self.set_text("",""), app.set_null("other_disease")}
                input_filter: app.block_typing
                 
            MDTextField:
                hint_text: "cardiovascular" 
                id: cardiovascular
                on_focus: if self.focus: {app.cardiovascular_menu.open(), self.set_text("",""), app.set_null("cardiovascular")}
                input_filter: app.block_typing
                 
            MDTextField:
                hint_text: "obesity"  
                id: obesity
                on_focus: if self.focus: {app.obesity_menu.open(), self.set_text("",""), app.set_null("obesity")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "renal chronic"  
                id: renal_chronic
                on_focus: if self.focus: {app.renal_chronic_menu.open(), self.set_text("",""), app.set_null("renal_chronic")}
                input_filter: app.block_typing
                
            MDTextField:
                hint_text: "tobacco"  
                id: tobacco
                on_focus: if self.focus: {app.tobacco_menu.open(), self.set_text("",""), app.set_null("tobacco"), app.set_null("tobacco")}   
                input_filter: app.block_typing 
                
            MDTextField:
                hint_text: "contact other covid" 
                id: contact_other_covid
                on_focus: if self.focus: {app.contact_other_covid_menu.open(), self.set_text("",""), app.set_null("contact_other_covid")} 
                input_filter: app.block_typing
                              
                
                          
    MDRaisedButton:
        text: "next"
        md_bg_color: 128/255,128/255,128/255,1
        pos_hint: {"center_x": 0.5, "center_y": 0.1}
        size_hint_x: 0.3
        size_hint_y: 0.065
        on_release: app.check_form() 
        
        
        
<ThirdScreen>:

    MDLabel:
        font_style: "H5"
        color: 248/255,248/255,255/255,1
        text: "Summary"
        halign: "center"
        pos_hint: {"center_x": 0.5, "center_y": 0.9}
        size_hint_x: 0.5
        size_hint_y: 0.5

    MDSlider:
        id: prediction_slider
        min: 0
        max: 4
        step: 1
        value: 1
        orientation: 'vertical'
        hint_text: "ofer"
        hint: False
        active: False
        show_off: False
        disabled: True
        
        color: 128/255,128/255,128/255,1
        pos_hint: {"center_x": 0.1, "center_y": 0.625}
        size_hint_x: 1
        size_hint_y: 0.35
        
        on_touch_down: app.slider_block(self)
        on_touch_up: app.slider_release(self)
   
        
    BoxLayout:
        id: prediction_slider_label     
        orientation: "vertical"
        pos_hint: {"center_x": 0.52, "center_y": 0.624}
        size_hint_x: 0.68
        size_hint_y: 0.45

        MDLabel:
            id: high_risk
            color: 255/255,0/255,0/255,1
            font_style: 'Body2'
        MDLabel:
            id: medium_risk
            color: 255/255,255/255,0/255,1
            font_style: 'Body2'
        MDLabel:
            id: low_risk
            color: 0/255,128/255,0/255,1
            font_style: 'Body2'
    
    MDIconButton:
        icon: "chart-gantt"
        user_font_size: "64sp"
        color: 128/255,128/255,128/255,1        
        pos_hint: {"center_x": 0.5, "center_y": 0.275}
        on_release: root.manager.current = 'FourthScreen'
     
           
    MDRaisedButton:
        text: "exit"
        md_bg_color: 128/255,128/255,128/255,1
        pos_hint: {"center_x": 0.5, "center_y": 0.1}
        size_hint_x: 0.3
        size_hint_y: 0.065
        on_release: app.clear_data()   
            
    
    
<FourthScreen>:  
    md_bg_color: 255/255,255/255,255/255,1
    on_touch_up: root.manager.current = 'ThirdScreen'   
    
    Image: 
        id: lime_report_img
        source: 'lime_report.jpg'
        pos_hint: {"center_x": 0.5, "center_y": 0.45}
        size_hint_x: 1.075
        size_hint_y: 1.075
                   
""")

class FirstScreen(MDScreen):
    pass

class SecondScreen(MDScreen):
    pass

class ThirdScreen(MDScreen):
    pass


class FourthScreen(MDScreen):
    pass

class CoronaApp(MDApp):
    debug_mode = False

    data = {
     "sex": "null",
     "HSD": "null",
     "entry_month": "null",
     "symptoms_month": "null",
     "pneumonia": "null",
     "age": "null",
     "pregnancy": "null",
     "diabetes": "null",
     "copd": "null",
     "asthma": "null",
     "inmsupr": "null",
     "hypertension": "null",
     "other_disease": "null",
     "cardiovascular": "null",
     "obesity": "null",
     "renal_chronic": "null",
     "tobacco": "null",
     "contact_other_covid": "null",
    }


    def build(self):
        self.title = "CoronaApp"
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = 'Green'
        self.theme_cls.accent_palette = 'Blue'
        self.theme_cls.theme_style = 'Dark'

        Window.size = (1080/3, 1920/3)
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(FirstScreen(name='FirstScreen'))
        sm.add_widget(SecondScreen(name='SecondScreen'))
        sm.add_widget(ThirdScreen(name='ThirdScreen'))
        sm.add_widget(FourthScreen(name='FourthScreen'))
        self.sm = sm

        self.init_menus()
        return sm

    def init_menus(self):
        sex_option = [{"text": "Male"}, {"text": "Female"}]
        self.sex_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.sex,
            items=sex_option,
            position="auto",
            width_mult=1.75,
        )
        self.sex_menu.bind(on_release=self.set_sex)

        month_option = [{"text": "January"}, {"text": "February"}, {"text": "March"}, {"text": "April"}, {"text": "May"}, {"text": "June"}, {"text": "July"}, {"text": "August"}, {"text": "September"}, {"text": "October"}, {"text": "September"}, {"text": "November"}, {"text": "December"}]
        self.entry_month_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.entry_month,
            items=month_option,
            position="auto",
            width_mult=2.15,
            border_margin='45dp'
        )
        self.entry_month_menu.bind(on_release=self.set_entry_month)
        self.symptoms_month_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.symptoms_month,
            items=month_option,
            position="auto",
            width_mult=2.15,
            border_margin='45dp'
        )
        self.symptoms_month_menu.bind(on_release=self.set_symptoms_month)

        binary_option = [{"text": "Yes"}, {"text": "No"}]
        self.pneumonia_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.pneumonia,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.pneumonia_menu.bind(on_release=self.set_pneumonia)
        self.pregnancy_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.pregnancy,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.pregnancy_menu.bind(on_release=self.set_pregnancy)
        self.diabetes_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.diabetes,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.diabetes_menu.bind(on_release=self.set_diabetes)
        self.copd_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.copd,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.copd_menu.bind(on_release=self.set_copd)
        self.asthma_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.asthma,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.asthma_menu.bind(on_release=self.set_asthma)
        self.inmsupr_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.inmsupr,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.inmsupr_menu.bind(on_release=self.set_inmsupr)
        self.hypertension_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.hypertension,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.hypertension_menu.bind(on_release=self.set_hypertension)
        self.other_disease_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.other_disease,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.other_disease_menu.bind(on_release=self.set_other_disease)
        self.cardiovascular_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.cardiovascular,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.cardiovascular_menu.bind(on_release=self.set_cardiovascular)
        self.obesity_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.obesity,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.obesity_menu.bind(on_release=self.set_obesity)
        self.renal_chronic_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.renal_chronic,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.renal_chronic_menu.bind(on_release=self.set_renal_chronic)
        self.tobacco_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.tobacco,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.tobacco_menu.bind(on_release=self.set_tobacco)
        self.contact_other_covid_menu = MDDropdownMenu(
            caller=self.sm.get_screen('SecondScreen').ids.contact_other_covid,
            items=binary_option,
            position="auto",
            width_mult=1.4,
        )
        self.contact_other_covid_menu.bind(on_release=self.set_contact_other_covid)



    def set_sex(self, instance_menu, instance_menu_item):
        def set_sex(interval):
            self.sm.get_screen('SecondScreen').ids.sex.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("sex",instance_menu_item.text)
        Clock.schedule_once(set_sex, 0.5)

    def set_entry_month(self, instance_menu, instance_menu_item):
        def set_entry_month(interval):
            self.sm.get_screen('SecondScreen').ids.entry_month.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("entry_month",instance_menu_item.text)
        Clock.schedule_once(set_entry_month, 0.5)

    def set_symptoms_month(self, instance_menu, instance_menu_item):
        def set_symptoms_month(interval):
            self.sm.get_screen('SecondScreen').ids.symptoms_month.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("symptoms_month",instance_menu_item.text)
        Clock.schedule_once(set_symptoms_month, 0.5)

    def set_pneumonia(self, instance_menu, instance_menu_item):
        def set_pneumonia(interval):
            self.sm.get_screen('SecondScreen').ids.pneumonia.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("pneumonia",instance_menu_item.text)
        Clock.schedule_once(set_pneumonia, 0.5)

    def set_pregnancy(self, instance_menu, instance_menu_item):
        def set_pregnancy(interval):
            self.sm.get_screen('SecondScreen').ids.pregnancy.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("pregnancy",instance_menu_item.text)
        Clock.schedule_once(set_pregnancy, 0.5)

    def set_diabetes(self, instance_menu, instance_menu_item):
        def set_diabetes(interval):
            self.sm.get_screen('SecondScreen').ids.diabetes.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("diabetes",instance_menu_item.text)
        Clock.schedule_once(set_diabetes, 0.5)

    def set_copd(self, instance_menu, instance_menu_item):
        def set_copd(interval):
            self.sm.get_screen('SecondScreen').ids.copd.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("copd",instance_menu_item.text)
        Clock.schedule_once(set_copd, 0.5)

    def set_asthma(self, instance_menu, instance_menu_item):
        def set_asthma(interval):
            self.sm.get_screen('SecondScreen').ids.asthma.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("asthma",instance_menu_item.text)
        Clock.schedule_once(set_asthma, 0.5)

    def set_inmsupr(self, instance_menu, instance_menu_item):
        def set_inmsupr(interval):
            self.sm.get_screen('SecondScreen').ids.inmsupr.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("inmsupr",instance_menu_item.text)
        Clock.schedule_once(set_inmsupr, 0.5)

    def set_hypertension(self, instance_menu, instance_menu_item):
        def set_hypertension(interval):
            self.sm.get_screen('SecondScreen').ids.hypertension.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("hypertension",instance_menu_item.text)
        Clock.schedule_once(set_hypertension, 0.5)

    def set_other_disease(self, instance_menu, instance_menu_item):
        def set_other_disease(interval):
            self.sm.get_screen('SecondScreen').ids.other_disease.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("other_disease",instance_menu_item.text)
        Clock.schedule_once(set_other_disease, 0.5)

    def set_cardiovascular(self, instance_menu, instance_menu_item):
        def set_cardiovascular(interval):
            self.sm.get_screen('SecondScreen').ids.cardiovascular.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("cardiovascular",instance_menu_item.text)
        Clock.schedule_once(set_cardiovascular, 0.5)

    def set_obesity(self, instance_menu, instance_menu_item):
        def set_obesity(interval):
            self.sm.get_screen('SecondScreen').ids.obesity.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("obesity",instance_menu_item.text)
        Clock.schedule_once(set_obesity, 0.5)

    def set_renal_chronic(self, instance_menu, instance_menu_item):
        def set_renal_chronic(interval):
            self.sm.get_screen('SecondScreen').ids.renal_chronic.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("renal_chronic",instance_menu_item.text)
        Clock.schedule_once(set_renal_chronic, 0.5)

    def set_tobacco(self, instance_menu, instance_menu_item):
        def set_tobacco(interval):
            self.sm.get_screen('SecondScreen').ids.tobacco.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("tobacco",instance_menu_item.text)
        Clock.schedule_once(set_tobacco, 0.5)

    def set_contact_other_covid(self, instance_menu, instance_menu_item):
        def set_contact_other_covid(interval):
            self.sm.get_screen('SecondScreen').ids.contact_other_covid.text = instance_menu_item.text
            instance_menu.dismiss()
            self.set_data("contact_other_covid",instance_menu_item.text)
        Clock.schedule_once(set_contact_other_covid, 0.5)

    def block_typing(self,str,u):
        return ""

    def unsigned_int(self,str,u):
        if str.isnumeric():
            return str
        else:
            return ""

    def set_null(self, str):
        self.data[str]="null"
        if self.debug_mode: print("data." + str + " = " +self.data[str])

    def set_data(self, id, option):
        if option == "Male":
            self.data[id] = "1"
        elif option == "Female":
            self.data[id] = "0"
        elif option == "Yes":
            self.data[id] = "1"
        elif option == "No":
            self.data[id] = "0"
        elif option == "January":
            self.data[id] = "1"
        elif option == "February":
            self.data[id] = "2"
        elif option == "March":
            self.data[id] = "3"
        elif option == "April":
            self.data[id] = "4"
        elif option == "May":
            self.data[id] = "5"
        elif option == "June":
            self.data[id] = "6"
        elif option == "July":
            self.data[id] = "7"
        elif option == "August":
            self.data[id] = "8"
        elif option == "September":
            self.data[id] = "9"
        elif option == "October":
            self.data[id] = "10"
        elif option == "November":
            self.data[id] = "11"
        elif option == "December":
            self.data[id] = "12"
        elif option == "December":
            self.data[id] = "12"
        elif option.isnumeric():
            if id == 'age':
                if (int(option) >= 130):
                    self.sm.get_screen('SecondScreen').ids.age.text = self.sm.get_screen('SecondScreen').ids.age.text[:-1]
                    return
                self.data[id] = math.floor(int(option) / 10)
            elif id == 'HSD':
                if (int(option) >= 100):
                    self.sm.get_screen('SecondScreen').ids.HSD.text = self.sm.get_screen('SecondScreen').ids.HSD.text[:-1]
                    return
                self.data[id] = option
            else: print("id error")
        else: self.data[id] = "null"

        if self.debug_mode: print("data." + id + " = " + str(self.data[id]))

    def check_form(self):
        if self.debug_mode: print(self.data)
        for input in self.data.values():
            if input == 'null':
                self.dialog = MDDialog(
                    title="Warning",
                    text="Please enter all data in the designated places",
                )
                self.dialog.open()
                if self.debug_mode: self.set_data_debug('1')
                return

        self.build_model_results()


    def build_model_results(self):
        (prediction, levels_number) = self.xai_model(list(self.data.values()))
        self.sm.get_screen('FourthScreen').ids.lime_report_img.reload()

        prediction_slider = self.sm.get_screen('ThirdScreen').ids.prediction_slider
        prediction_slider.value = prediction / (levels_number-1) * 4

        if levels_number == 3 or levels_number == 5:
            self.sm.get_screen('ThirdScreen').ids.high_risk.text = "High risk:" + "\n" + "The patient will probably need the help of intubation."
            self.sm.get_screen('ThirdScreen').ids.medium_risk.text = "Medium risk:" + "\n" + "The patient will probably need to be hospitalized at ICU."
            self.sm.get_screen('ThirdScreen').ids.low_risk.text = "Low risk:" + "\n" + "The patient will probably not need hospitalization."
        elif levels_number == 2:
            self.sm.get_screen('ThirdScreen').ids.prediction_slider.size_hint_y = 0.21
            self.sm.get_screen('ThirdScreen').ids.prediction_slider_label.size_hint_y = 0.25
            self.sm.get_screen(
                'ThirdScreen').ids.high_risk.text = "High risk:" + "\n" + "The patient will probably need to be hospitalized in ICU and may need help of intubation."
            self.sm.get_screen(
                'ThirdScreen').ids.low_risk.text = "Low risk:" + "\n" + "The patient will probably only need home isolation or regular hospitalization for follow-up."
        else:
            if self.debug_mode: print("levels_number error")

        self.sm.current = 'ThirdScreen'

    def xai_model(self,ts):
        ts = np.array(ts).astype(np.float)
        if self.debug_mode: print(list(self.data.values()))

        features_names = ['sex', 'HSD', 'entry_month', 'symptoms_month', 'pneumonia', 'age_group', 'pregnancy',
                          'diabetes',
                          'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
                          'renal_chronic', 'tobacco', 'contact_other_covid']
        ls = []
        model = load_model('model')
        with open('config.X', 'rb') as config_X_file:
            X = pickle.load(config_X_file)

        ls.append(ts)
        ls = np.array(ls)

        prediction = model.predict_classes(ls)
        prediction = int(prediction[0])
        if self.debug_mode: print('prediction = ' + str(prediction))
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=features_names,
                                                           class_names=['ill_level'],
                                                           verbose=False, mode='regression')
        exp = explainer.explain_instance(ls[0], model.predict, num_features=len(features_names))
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(18, 9)
        fig.savefig(f'lime_report.jpg', dpi=100)
        fig = Image.open("lime_report.jpg")
        fig = fig.transpose(Image.ROTATE_270)
        fig.save("lime_report.jpg", "JPEG")

        return (prediction,2)

    def slider_block(self, slider):
        slider.disabled = True

    def slider_release(self, slider):
        slider.disabled = False

    def clear_data(self):
        for text_input in self.sm.get_screen('SecondScreen').ids.form.children:
            text_input.text = ''
        self.data = dict.fromkeys(self.data,'null')
        self.sm.current = 'FirstScreen'

    def set_data_debug(self, val):
        self.data = dict.fromkeys(self.data, val)

if __name__ == "__main__":
    CoronaApp().run()

