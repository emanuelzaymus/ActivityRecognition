class Aruba():
    __FILE = 'data/data_aruba_formatted_5days.txt'
    __ACTIVITIES = ['Meal_Preparation', 'Relax', 'Eating', 'Work', 'Sleeping', 'Wash_Dishes', 'Bed_to_Toilet',
                    'Enter_Home', 'Leave_Home', 'Housekeeping', 'Resperate', '!']
    __NUM_ACTIVITIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    @property
    def file(self) -> str:
        return self.__FILE
