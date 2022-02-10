//
//  ProfileCreationView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 24.01.21.
//

import SwiftUI



enum SourceType {
    case photoLibrary
    case camera
}

struct ProfileCreationView: View {
    
    
    @StateObject private var addProfileCreationVM = ProfileCreationModel()
    @StateObject private var firestoreFotoMangerUserTest = FirestoreFotoManagerUserTest()
    
    // binding to show the main controller that the proccess is finished
    @Binding var profileCreationFinished: Bool
    
    @Binding var userToken: String
    
    // MARK: - state vars
    
    // var for the birthday date of the profile owner in the date format
    @State var birthdayDate: String = "01/01/1970"
    
    
    // user accepts the location usage - NEEDS TO BE CHANGED TO BOOL
    @State var acceptLocation: String = "True"
    
    // picture text which will change when picture is uploaded - does not need to go into the database
    @State var pictureText: String = "Picture upload"
    
    // show the alertbox
    @State var showAlertBox = false
    
    // state var to catch what the user has handed into the textfield of the alertbox
    @State var outputAlertBox = ""
    
    // image handling
    
    // show
    @State var showImagePicker = false
    // images in array
    @State var images: [UIImage] = []
    @State var croppedImage: UIImage?
    // temp image which is used with the new cropper
    @State private var tempInputImage: UIImage?
    
    // image (swiftUI)
    //@State var image: Image?
    // image UIKit
    //@State var uiImage: UIImage?
    
    // var to check at which step the user wants to hand in data
    @State var pathwayStep = 0
    
    // icon name changes after the value is set
    @State var iconName = "pencil.circle"
    
    // background color changes after the value is set
    @State var backgroundColor = "BackgroundMain"
    
    // var to see if the action in the alert was accepted or not
    @State var acceptedAction = [false, false, false, false, false, false]
    @State var updateButtonTapped: Bool = false
    
    var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.timeZone = TimeZone(abbreviation: "GMT+0:00")
        formatter.dateFormat = "dd/MM/yy"
        return formatter
    }()
    
    
    var body: some View {
        ZStack {
            
            // box where all items are in with fixed size
            ZStack {
                Color.clear
                    .frame(width: 340, height: 620, alignment: .center)
                    .modifier(FrozenWindowModifier())
                
                VStack {
                    VStack {
                        Text("Let's follow the completion path!")
                            .font(.largeTitle)
                            .padding(.top, 16)
                    }
                    ScrollView {
                        
                        // view to fill in the name
                        VStack {
                            NameLineView(name: $addProfileCreationVM.name, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[0] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[0] ? .constant("Clear") : .constant("BackgroundMain"))
                            
                            // create image for the first to second step pathway
                            Image("Pathway-ProfileCreation")
                                .resizable()
                                .frame(width: 268.58, height: 92.92, alignment: .center)
                            
                            // get the gender of the user
                            GenderLineView(gender: $addProfileCreationVM.gender, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[1] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[1] ? .constant("Clear") : .constant("BackgroundMain"))
                            
                            // create image for the second to the third step of the pathway
                            Image("Pathway-ProfileCreation")
                                .resizable()
                                .frame(width: 268.58, height: 92.92, alignment: .center)
                                .rotation3DEffect(
                                    Angle(degrees: 180),
                                    axis: (x: 0, y: 1.0, z: 0.0)
                                )
                        }
                        
                        // get the birthday date of the user
                        BirthdayLineView(birthday: $birthdayDate, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[2] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[2] ? .constant("Clear") : .constant("BackgroundMain"))
                        
                        //                        // create image for the third to the fourth step of the pathway
                        //                        Image("Pathway-ProfileCreation")
                        //                            .resizable()
                        //                            .frame(width: 268.58, height: 92.92, alignment: .center)
                        //
                        //                        // get for what the user is searching (women, men, All)
                        //                        SearchingLineView(searchingFor: $addProfileCreationVM.searchingFor, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[3] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[3] ? .constant("Clear") : .constant("BackgroundMain"))
                        
                        // create image for the fifth to the sixth step of the pathway
                        Image("Pathway-ProfileCreation")
                            .resizable()
                            .frame(width: 268.58, height: 92.92, alignment: .center)
                        
                        //                        // get the user permission to use the current position
                        //                        LocationLineView(acceptLocation: $acceptLocation, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[4] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[4] ? .constant("Clear") : .constant("BackgroundMain"))
                        //
                        //                        // create image for the fourth to the fifth step of the pathway
                        //                        Image("Pathway-ProfileCreation")
                        //                            .resizable()
                        //                            .frame(width: 268.58, height: 92.92, alignment: .center)
                        //                            .rotation3DEffect(
                        //                                Angle(degrees: 180),
                        //                                axis: (x: 0, y: 1.0, z: 0.0)
                        //                            )
                        
                        // upload image to database
                        PictureLineView(pictureText: acceptedAction[5] ? .constant("You look good today") : .constant("Profile picture"), showPicker: $showImagePicker, pathwayStep: $pathwayStep, showAlertBox: $showAlertBox, iconName: acceptedAction[5] ? .constant("checkmark.circle") : .constant("pencil.circle"), backgroundColor: acceptedAction[5] ? .constant("Clear") : .constant("BackgroundMain"))
                        
                        // create image for the end pathway to the update profile button
                        VStack(alignment: .trailing) {
                            
                            HStack {
                                Spacer()
                                Image("Endway-ProfileCreation")
                                    .resizable()
                                    .frame(width: 137.58, height: 47.6, alignment: .trailing)
                            }
                            
                        }
                        .frame(width: 268.58)
                        
                        // button is disabled until the user set at least the gender
                        let enoughInformation = acceptedAction[1]
//                            && acceptedAction[1] && acceptedAction[2]
                        
//                        if updateButtonTapped && !acceptedAction[0] {
//                            Text("Please enter your name at the very top")
//                                .font(.subheadline)
//                                .foregroundColor(.accentColor)
                        if updateButtonTapped && !acceptedAction[1] {
                            Text("Please enter your gender")
                                .font(.subheadline)
                                .foregroundColor(.accentColor)
                        }
//                        } else if updateButtonTapped && !acceptedAction[2] {
//                            Text("Please enter your birthdate")
//                                .font(.subheadline)
//                                .foregroundColor(.accentColor)
//                        }
                        
                        // MARK: update button
                        Button(action: {
                            
                            updateButtonTapped = true
                            
                            if enoughInformation {
                                
                                if !acceptedAction[0] {
                                    addProfileCreationVM.name = "Nameless"
                                }
                                
                                if !acceptedAction[2] {
                                    // is not used idk
                                    addProfileCreationVM.birthdayDate = Date() - 568025136
//                                    addProfileCreationVM.birthdayDate = "01/01/1970"
                                }
                                
                                addProfileCreationVM.birthdayDate = dateFormatter.date(from: birthdayDate) ?? (Date() - 568025136)
                                
                                addProfileCreationVM.userToken = userToken  
                                
                                addProfileCreationVM.searchingFor = "All"
                                
                                addProfileCreationVM.createUser(images: [croppedImage ?? UIImage(named: "FemaleStockImage")!])
                                // haptic feedback when button is tapped
                                hapticPulse(feedback: .rigid)
                                
                                profileCreationFinished = true
                            }
                            
                        }, label: {
                            HStack {
                                Text("Update profile")
                                    .foregroundColor(.primary)
                                Image(systemName: "checkmark.circle")
                                
                            }
                            .padding()
                            .modifier(FrozenWindowModifier())
                        })
                        
                        .padding(.bottom, 16)
                        .opacity(enoughInformation ? 1 : 0.7)
                        
                        .onChange(of: addProfileCreationVM.saved, perform: { value in
                            if value {
                                //if save successful
                                profileCreationFinished = true
                            } else {
                                //if profile settings didn't upload
                            }
                        })
                        
                        
                        
                    }
                    .padding(.bottom, 16)
                    
                }
                .padding(.horizontal, 16)
                // bring the content VStack to the same size as the background shade
                .frame(width: 340, height: 620, alignment: .center)
                
                
                // creating pop-up alert-boxes
                if showAlertBox {
                    
                    // switch case to show the correct alertbox for each step in the pathway
                    switch pathwayStep {
                    
                    // case 0 is the first step -> name creation
                    case 0:
                        AlertBoxView(title: "Type in your Name", placeholder: "Type here..", defaultText: "Name", textFieldInput: true, selectedDuration: .constant(.medium), output: $addProfileCreationVM.name, show: $showAlertBox, accepted: $acceptedAction[0])
                            // z index 1 == the top layer -> this is needed due to animation processes
                            .zIndex(1.0)
                        
                    // case 1 is the second step -> gender creation
                    case 1:
                        AlertBoxView(title: "Choose your gender", placeholder: "Tap here to choose..", defaultText: "Gender", pickerInput: true, selectedDuration: .constant(.medium), pickerInputArray: ["Male", "Female", "Other"], output: $addProfileCreationVM.gender, show: $showAlertBox, accepted: $acceptedAction[1])
                            .zIndex(1.0)
                        
                    // case 2 is the third step -> birthday date
                    case 2:
                        AlertBoxView(title: "Select your birthday date", placeholder: "Tap here to choose..", defaultText: "Birthday", selectedDuration: .constant(.medium), dateInput: true, output: $birthdayDate, show: $showAlertBox, accepted: $acceptedAction[2])
                            .zIndex(1.0)
                        
                    // case 3 is the fourth step -> searching for creation
                    case 3:
                        AlertBoxView(title: "Choose for whom you are searching", placeholder: "Tap here to choose..", defaultText: "Searching for", pickerInput: true, selectedDuration: .constant(.medium), pickerInputArray: ["Male", "Female", "All"], output: $addProfileCreationVM.searchingFor, show: $showAlertBox, accepted: $acceptedAction[3])
                            .zIndex(1.0)
                        
                    case 4:
                        // add alert-box to ask user for location services
                        AlertBoxView(title: "Allow app to use your current location", placeholder: "", defaultText: "", selectedDuration: .constant(.medium), output: $acceptLocation, show: $showAlertBox, accepted: $acceptedAction[4])
                            .zIndex(1.0)
                        
                    // image picker view
                    case 5:
                        Color.clear
                            .sheet(isPresented: $showImagePicker, content: {
                                ImageHandler(croppedImage: $croppedImage, showView: $showImagePicker)
                            })
                            .onChange(of: croppedImage, perform: { value in
                                acceptedAction[5] = true
                            })
                    
                    // the default is 0 which is the first step in the pathway -> name creation
                    default:
                    AlertBoxView(title: "Type in your Name", placeholder: "Type here..", defaultText: "Name", selectedDuration: .constant(.medium), output: $outputAlertBox, show: $showAlertBox, accepted: $acceptedAction[2])
                    // z index 1 == the top layer -> this is needed due to animation processes
                    .zIndex(1.0)
                }
                
            }
            
            
        }
        .animation(.spring(blendDuration: 0.15))
        
    }
}

}

struct ProfileCreationView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileCreationView(profileCreationFinished: .constant(false), userToken: .constant(""))
        //            .previewDevice("iPhone 8")
    }
}

struct NameLineView: View {
    
    // Binding from main View
    @Binding var name: String
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 0
                showAlertBox = true
            }) {
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                
                Text(name)
                    .foregroundColor(.primary)
            }
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .leading)
    }
}

struct GenderLineView: View {
    
    // Binding from main View
    @Binding var gender: String
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            
            
            
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 1
                showAlertBox = true
            }) {
                Text(gender)
                    .foregroundColor(.primary)
                
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .trailing)
    }
}

struct BirthdayLineView: View {
    
    // Binding from main View
    @Binding var birthday: String
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 2
                showAlertBox = true
            }) {
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                
                Text(birthday)
                    .foregroundColor(.primary)
                    .onAppear {
                        birthday = "Birthday"
                    }
            }
            
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .leading)
    }
}



struct SearchingLineView: View {
    
    // Binding from main View
    @Binding var searchingFor: String
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            
            Text(searchingFor)
            
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 3
                showAlertBox = true
            }) {
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .trailing)
    }
}

struct LocationLineView: View {
    
    // Binding from main View
    @Binding var acceptLocation: String
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 4
                showAlertBox = true
            }) {
                
                Text("Location")
                    .foregroundColor(.primary)
                
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .leading)
    }
}

struct PictureLineView: View {
    
    // Binding from main View
    @Binding var pictureText: String
    
    // show imagePicker
    @Binding var showPicker: Bool
    
    // binding for pathway change
    @Binding var pathwayStep: Int
    
    // binding for show/hide of alertBox
    @Binding var showAlertBox: Bool
    
    // icon name changes after the value is set
    @Binding var iconName: String
    
    // background color changes after the value is set
    @Binding var backgroundColor: String
    
    var body: some View {
        HStack {
            
            
            
            Button(action: {
                // configure which step it is in the pathway
                pathwayStep = 5
                showPicker = true
                showAlertBox = true
                
            }) {
                Image(systemName: iconName)
                    .font(.title)
                    .padding(4)
                    .background(Color(backgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                
                Text(pictureText)
                    .foregroundColor(.primary)
            }
            
            
        }
        .padding(.horizontal, 16)
        .frame(width: 340, alignment: .trailing)
    }
}

