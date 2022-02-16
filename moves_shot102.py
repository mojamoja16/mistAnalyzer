import tkinter
import serial 
from time import sleep



class shot102():
    def __init__(self):
        #threading.Thread.__init__(self)
        self._pulse_per_unit =(1000,1000)
        self._ser = serial.Serial('COM18')
        self._ser.baudrate = 38400
        self._write_waittime = 0.05
        #self._ser.write(b"!:\r\n")
        sleep(self._write_waittime)
        

    def clear_buf(self):
        self._ser.write("\r\n".encode())
        
        sleep(self._write_waittime)
        self._ser.read_all()
    
    def read_retval(self):
        sleep(self._write_waittime)
        read_val=self._ser.read_all().decode().splitlines()[0]
        return read_val
    
    def is_busy(self):

        self.clear_buf()
        self._ser.write(b"!:\r\n")
        status = self.read_retval()

        if status == "B":
            print("busy")
            return True
        else:
            print("read")
            return False
    
    def hold_motor(self,axis):
        self.clear_buf()
        sending_command="C:{}1\r\n".format(axis).encode()
        self._ser.write(sending_command)
        status = self.read_retval()
        return status

    def release_motor(self,axis):
        self.clear_buf()
        sending_command = "C:{}0\r\n".format(axis).encode()
        self._ser.write(sending_command)
        status= self.read_retval()
        self.set_cmd_absolute_move_unit(2, -2)
        return status

    def go_mechanical_origin(self,axis,direction):
        """
        機械原点復帰
        :param axis: 軸番号（1, 2)
        :param direction: 軸方向（'+' or '-')
        :return: status
        """

        self.clear_buf()
        sending_command="H:{}{}\r\n".format(axis,direction).encode()
        self._ser.write(sending_command)
        status = self.read_retval()

        while self.is_busy():
            sleep(0.1)
        return status

    def set_curent_position_as_orijin(self,axis):
        self.clear_buf()
        sending_command = "R:{}\r\n".format(axis).encode()
        self._ser.write(sending_command)
        status =self.read_retval

        while self.is_busy():
            sleep(0,1)
        return status

    def set_cmd_relative_move_pulse(self, axis, n_pulse):
        self.clear_buf()
        if n_pulse > 0:
            direction = "+"
        else:
            direction = "-"
        sending_command = "M:{}{}P{}\r\n".format(axis,direction,abs(n_pulse)).encode()
        self._ser.write(sending_command)
        
        status = self.read_retval()
        return status

    def set_cmd_absolute_move_pulse(self,axis,n_pulse):
        self.clear_buf()
        if n_pulse > 0:
            direction = "+"

        else:
            direction = "-"
        sending_command = "A:{}{}P{}\r\n".format(axis,direction,abs(n_pulse)).encode()
        self._ser.write(sending_command)
        status = self.read_retval()
        return status

    def unit_to_pulse(self,axis,unit_val):
        return int(unit_val*self._pulse_per_unit[axis-1])

    def set_cmd_relative_move_unit(self,axis,unit_val):
        n_pulse = self.unit_to_pulse(axis,unit_val)
        status = self.set_cmd_relative_move_pulse(axis,n_pulse)
        return status

    def set_cmd_absolute_move_unit(self,axis,unit_val):
        n_pulse = self.unit_to_pulse(axis,unit_val)
        status =self.set_cmd_absolute_move_pulse(axis,n_pulse)
        print("set_cmd_absolute")
        return status

    def start_move(self):
        self.clear_buf()
        sending_command = "G:\r\n".encode()
        self._ser.write(sending_command)
        status=self.read_retval()

        while self.is_busy():
            sleep(0.1)
        return status
    
class Model():
    def __init__(self,app,shot):
        self.master = app
        self.shot=shot
        self.create_item()
        self.create_buttons()
        self.create_textbox()

    def create_item(self):
        canvas_width=300
        canvas_height=200

        self.main_frame = tkinter.Frame(self.master,bg="gray15")
        self.main_frame.pack()

        self.operation_frame = tkinter.Frame(self.main_frame,bg="gray15")
        self.operation_frame.grid(column=1,row=1)

    def create_buttons(self):

        self.start_move_button=tkinter.Button(
            self.operation_frame,
            text="start"
        )
        self.start_move_button.pack()

        self.stop_move_button=tkinter.Button(
            self.operation_frame,
            text="stop"
        )
        self.stop_move_button.pack()
        
    def create_textbox(self):
        self.move_pulse_text=tkinter.Text(self.operation_frame,
                                          width=5,
                                          height=1,
                                          bd=1,)
        self.move_pulse_text.pack()

class Main():
    def __init__(self,app,model,shot):
        self.master=app
        self.model=model
        self.shot=shot

        #shot-102が動いているかどうかの管理
        self._is_moving = False

        self.move_direction =True

        self.timer=1000
        self.set_events()

    def set_events(self):
        self.model.start_move_button['command']=self.push_start_move_button
        self.model.stop_move_button['command']=self.push_stop_move_button

    def push_start_move_button(self):

        if self._is_moving==False:
            self.master.after(self.timer,self.move_shot)
            self._is_moving = True

    def push_stop_move_button(self):
        self._is_moving =False
        self.shot._ser.close()

    def move_shot(self):
        if self._is_moving == False:
            pass

        self.master.after(self.timer,self.move_shot)
        sleep(0.1)

        if self.move_direction==True:
            self.shot.set_cmd_relative_move_pulse(1, -3000)
            self.shot.start_move()
            self.move_direction=False
        else:
            self.shot.set_cmd_relative_move_pulse(1, +3000)
            self.shot.start_move()
            self.move_direction=True

app = tkinter.Tk()
app.title('shot102')
shot=shot102()
model= Model(app,shot)
main = Main(app,model,shot)
app.mainloop()
