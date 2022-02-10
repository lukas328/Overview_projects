//
//  AlertBoxView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 25.01.21.
//

import SwiftUI
import Combine

struct AlertBoxView: View {
    
    // MARK: - Parameter to configure the alert
    
    // title of alert box
    var title: String
    
    // placeholder for textfield
    var placeholder: String
    
    // default value to check if the user made correct input
    var defaultText: String
    
    // design the alertBox to have a textField input possibility
    var textFieldInput = false
    
    // design the alertBox to have a Picker input possibility
    var pickerInput = false
    
    // -- new
    
    // own categories with limit length
    var textFieldInputWithLimit = false
    @State var ownCategory: String = ""
    let textLimit = 10
    
    var categoryPicker = false
    @State var selectedCategory = Category.cafe
    
    var datePicker: Bool = false
    @State var selectedDate = Date()
    
    var startTime: String = ""
    let minMaxRange = Date() - 86400 ... Date() + 86400
    var timePicker: Bool = false
    @State var selectedTime = Date()
    
    var durationPicker: Bool = false
    @Binding var selectedDuration: Duration
    
    // get nearest quarter as start
    
    let roundedDate = { () -> Date in
        let now = Date()
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: now)
        let minute = calendar.component(.minute, from: now)

        // Round down to nearest date eg. every 15 minutes
        let minuteGranuity = 15
        let roundedMinute = minute - (minute % minuteGranuity)
        
        return calendar.date(bySettingHour: hour,
                             minute: roundedMinute,
                             second: 0,
                             of: now)!
    }()
    
    // -- new end
    
    // array which is shown in the picker view
    var pickerInputArray = [""]
    
    // design the alertBox to have a DatePicker input possibility
    var dateInput = false
    
    // design the date format of the alertbox
    var dateFormat = "dd/MM/YY"
    
    // cancel button per default
    var cancelButton = "Cancel"
    
    // confirmButton per default
    var confirmButton = "OK"
    
    // MARK: - Bindings
    
    // output of the alert input textfield
    @Binding var output: String
    
    // show/hide param
    @Binding var show: Bool
    
    // accepted param to give the information back
    @Binding var accepted: Bool
    
    // lastSlectedIndex is need to get the value out of the picker array in the picker mode of the alert box
    @State var lastSelectedIndex: Int?
    
    // date
    @State var date: Date = Date()
    
    // empty textField
    @State var emptyText: String = ""
    
    
    var body: some View {
        
        ZStack {
            
            // background
            BlurView(style: .systemUltraThinMaterial)
                .opacity(0.9)
                .ignoresSafeArea()
            
            // main window
            VStack {
                
                // title text
                VStack {
                    Text(title)
                        .font(.title2)
                        .bold()
                    
                    // only show textField if it defined
                    if textFieldInput {
                        TextField(placeholder, text: $emptyText)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .padding(.horizontal, 20)
                    }
                    
                    // show with limited textField
                    if textFieldInputWithLimit {
                        TextField("Other", text: $ownCategory)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .padding(.horizontal, 20)
                            .onReceive(Just(ownCategory)) { _ in limitText(textLimit) }
                    }
                    
                }
                // only show Picker if it defined
                if pickerInput {
                    // picker
                    PickerTextField(data: pickerInputArray, placerholder: placeholder, lastSelectedIndex: $lastSelectedIndex)
                        .frame(height: 35)
                        .padding(.horizontal, 20)
                }
                
                if categoryPicker {
                    Picker("This Title Is Localized", selection: $selectedCategory) {
                        ForEach(Category.allCases, id: \.self) { value in
                            Text(value.localizedName)
                                .tag(value)
                        }
                    }
                    
                    .pickerStyle(WheelPickerStyle())
                }
                
                if datePicker {
                    // pick a date from now to the future "..."
                    DatePicker("Please enter a date", selection: $selectedDate, in: Date()..., displayedComponents: .date)
                        .labelsHidden()
                }
                
                if timePicker {
                    // pick a startTime from now to the future "..."
                    DatePickerX(selection: $selectedTime, in: minMaxRange, minuteInterval: 15)
                        
                }
                
                if durationPicker {
                    Picker("This Title Is Localized", selection: $selectedDuration) {
                        ForEach(Duration.allCases, id: \.self) { value in
                            Text(value.localizedName)
                                .tag(value)
                        }
                    }                    
                    .pickerStyle(WheelPickerStyle())
                }
                
                // only show the DatePicker if it is defined
                if dateInput {
                    // date picker
                    let dateTextField = DateTextField(date: $date)
                    
                    // is change the dateFormat and returning a view
                    dateTextField.dateFormat(dateFormat)
                    
                }
                
                
                Divider()
                    .padding(.horizontal, 20)
                
                HStack {
                    
                    // button to cancel the action
                    Button(action: {
                        output = defaultText
                        withAnimation(.spring()) {
                            self.show.toggle()
                            
                            accepted = false
                        }
                    }) {
                        Text(cancelButton)
                            .frame(width: 118, height: 40)
                    }
                    
                    Divider()
                        .frame(height: 25)
                    
                    // button to confirm the action
                    Button(action: {
                        withAnimation(.spring()) {
                            self.show.toggle()
                            
                            if textFieldInput {
                                output = emptyText
                            }
                            
                            if textFieldInputWithLimit {
                                if ownCategory != "" {
                                    output = ownCategory
                                } else {
                                    output = "Other Event"
                                }
                                

                            }
                            
                            // check if the picker was used then the pickeroutput needs to be assigned to the normal output
                            if pickerInput {
                                output = pickerInputArray[lastSelectedIndex ?? 0]
                            }
                            
                            if categoryPicker {
                                output = selectedCategory.rawValue
                            }
                            
                            if datePicker {
                                let dateFormatter = DateFormatter()
                                dateFormatter.locale = Locale(identifier: "en_DE")
                                dateFormatter.dateFormat = self.dateFormat
                                output = dateFormatter.string(from: selectedDate)
                            }
                            
                            if timePicker {
                                let dateFormatter = DateFormatter()
                                dateFormatter.locale = Locale(identifier: "en_DE")
                                dateFormatter.dateFormat = self.dateFormat
                                output = dateFormatter.string(from: selectedTime)
                            }
                            
                            if durationPicker {
                                var dateOutput = Date()
                                let dateFormatter = DateFormatter()
                                dateFormatter.locale = Locale(identifier: "en_DE")
                                dateFormatter.dateFormat = self.dateFormat
                                dateOutput = dateFormatter.date(from: startTime) ?? Date()
                                
                                switch selectedDuration {
                                case .veryShort:
                                    dateOutput += 30 * 60
                                case .short:
                                    dateOutput += 45 * 60
                                case .medium:
                                    dateOutput += 60 * 60
                                case .normal:
                                    dateOutput += 90 * 60
                                case .long:
                                    dateOutput += 120 * 60
                                case .veryLong:
                                    dateOutput += 180 * 60
                                }
                                
                                output = dateFormatter.string(from: dateOutput)
                            }
                            
                            if dateInput {
                                let dateFormatter = DateFormatter()
                                dateFormatter.locale = Locale(identifier: "en_DE")
                                dateFormatter.dateFormat = self.dateFormat
                                output = dateFormatter.string(from: date)
                            }
                            
                            // check if output is useful
                            if output != defaultText && output != "" {
                                accepted = true
                            } else {
                                output = defaultText
                                accepted = false
                            }
                            
                        }
                        
                    })
                    {
                        Text(confirmButton)
                            .frame(width: 118, height: 40)
                    }
                }
            }
            .padding(8)
            .frame(width: 300, alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/)
            .modifier(FrozenWindowModifier())
        }
        .onAppear {
            selectedTime = roundedDate
        }
    }
    
    
    func getBirthdayDate() -> Date {
        return self.date
    }
    
    //Function to keep text length in limits
        func limitText(_ upper: Int) {
            if ownCategory.count > upper {
                ownCategory = String(ownCategory.prefix(upper))
            }
        }
}

struct AlertBoxView_Previews: PreviewProvider {
    static var previews: some View {
        AlertBoxView(title: "Alert", placeholder: "Text here..", defaultText: "Name", selectedDuration: .constant(.medium), output: .constant(""), show: .constant(true), accepted: .constant(false))
    }
}
