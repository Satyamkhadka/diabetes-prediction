import tkinter as tk
import joblib
input_window = tk.Tk()


gender=tk.IntVar()
polyuria=tk.IntVar()
polydipsia=tk.IntVar()
partial_paresis=tk.IntVar()
sudden_weight_loss=tk.IntVar()
itching=tk.IntVar()
weakness=tk.IntVar()
polyphagia=tk.IntVar()
genital_thrush=tk.IntVar()
visual_blurring=tk.IntVar()
obesity=tk.IntVar()
irritability=tk.IntVar()
delayed_healing=tk.IntVar()
muscle_stiffness=tk.IntVar()
output_label_var=tk.StringVar()


age_label = tk.Label(input_window, text="age")
age_c =tk.Entry(input_window, text = "age" ,bd=5, width = 20)
polyuria_c =tk.Checkbutton(input_window, text = "polyuria", variable =polyuria , onvalue = 1, offvalue = 0, height=2,width = 20)
polydipsia_c =tk.Checkbutton(input_window, text = "polydipsia", variable =polydipsia , onvalue = 1, offvalue = 0, height=2,width = 20)
# age_c =tk.Checkbutton(input_window, text = "age", variable =age , onvalue = 1, offvalue = 0, height=2,width = 20)
gender_c =tk.Checkbutton(input_window, text = "gender(check if male)", variable =gender , onvalue = 1, offvalue = 0, height=2,width = 20)
partial_paresis_c =tk.Checkbutton(input_window, text = "partial_paresis", variable =partial_paresis , onvalue = 1, offvalue = 0, height=2,width = 20)
sudden_weight_loss_c =tk.Checkbutton(input_window, text = "sudden_weight_loss", variable =sudden_weight_loss , onvalue = 1, offvalue = 0, height=2,width = 20)
weakness_c=tk.Checkbutton(input_window, text = "weakness", variable =weakness , onvalue = 1, offvalue = 0, height=2,width = 20)
polyphagia_c=tk.Checkbutton(input_window, text = "polyphagia", variable =polyphagia , onvalue = 1, offvalue = 0, height=2,width = 20)
genital_thrush_c=tk.Checkbutton(input_window, text = "genital_thrush", variable =genital_thrush , onvalue = 1, offvalue = 0, height=2,width = 20)
visual_blurring_c=tk.Checkbutton(input_window, text = "visual_blurring", variable =visual_blurring , onvalue = 1, offvalue = 0, height=2,width = 20)
obesity_c=tk.Checkbutton(input_window, text = "obesity", variable =obesity , onvalue = 1, offvalue = 0, height=2,width = 20)
irritability_c =tk.Checkbutton(input_window, text = "irritability", variable =irritability , onvalue = 1, offvalue = 0, height=2,width = 20)
delayed_healing_c =tk.Checkbutton(input_window, text = "delayed_healing", variable =delayed_healing , onvalue = 1, offvalue = 0, height=2,width = 20)
muscle_stiffness_c =tk.Checkbutton(input_window, text = "alopecia", variable =muscle_stiffness , onvalue = 1, offvalue = 0, height=2,width = 20)
itching_c =tk.Checkbutton(input_window, text = "itching", variable =itching , onvalue = 1, offvalue = 0, height=2,width = 20)
output_label = tk.Label(input_window, textvariable=output_label_var,text="Please Input the values")



def handle_submit(event):
   
    output_label_var.set("Loading!!!")
    output = []
    output.append(int(age_c.get())/100)
    output.append(gender.get())
    output.append(polyuria.get())
    output.append(polydipsia.get())
    output.append(sudden_weight_loss.get())
    output.append(weakness.get())
    output.append(polyphagia.get())
    output.append(genital_thrush.get())
    output.append(visual_blurring.get())
    output.append(obesity.get())
    output.append(itching.get())
    output.append(irritability.get())
    output.append(delayed_healing.get())
    output.append(partial_paresis.get())
    output.append(muscle_stiffness.get())

    print(output)
    predict = [output,]
    loaded_model = joblib.load('rf.joblib')
    print(loaded_model)
    result = loaded_model.predict([output])
    print("the resut is:")
    print(result[0])
    if result[0] == 0:
        output_label_var.set("Congratulations you have got no Diabetes!")
    else:
        output_label_var.set("You seem to have high probability of diabetes.")


button = tk.Button(text="Check diabetes!")
button.bind("<Button-1>", handle_submit)


polyuria_c.pack()
polydipsia_c.pack()
age_label.pack()
age_c.pack()
gender_c.pack()
partial_paresis_c.pack()
sudden_weight_loss_c.pack()
irritability_c.pack()
delayed_healing_c.pack()
muscle_stiffness_c.pack()
weakness_c.pack()
polyphagia_c.pack()
genital_thrush_c.pack()
visual_blurring_c.pack()
obesity_c.pack()
itching_c.pack()
button.pack()
output_label.pack()


input_window.mainloop()

