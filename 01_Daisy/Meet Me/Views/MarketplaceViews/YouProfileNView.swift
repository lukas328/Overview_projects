//
//  YouProfileNView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 12.03.21.
//

import SwiftUI
import URLImage

struct YouProfileNView: View {
    
    @StateObject var youProfileVM: YouProfilViewModel = YouProfilViewModel()
    
    @Binding var showYouProfileView: Bool
    @Binding var event: EventModel
    
    @State var showPictureIndex: Int = 0
    
    
    var body: some View {
        
        GeometryReader { bounds in
            ZStack {
                
                Color.black.opacity(0.001)
                    .onTapGesture(perform: {
                        print("area tapped - out of you profile")
                        showYouProfileView = false
                    })
                
                VStack {
                    
                    // spacer is used to get full area to tap
                    Spacer()
                    
                    
                    ZStack {
                        
                        // Base (including picture) of the Profile
                        base
                        
                        // information-box at the bottom of the base
                        informationBox
                            .offset(y: (bounds.size.width - 48) * 1.33 / 2)
                        
                        // show the event at the top right corner
                        eventCircle
                            .offset(x: (bounds.size.width - 48) / 2 - 30, y: -((bounds.size.width - 48) * 1.33) / 2 + 30)
                            // dismiss Profile when users taps on Event picture
                            .onTapGesture {
                                showYouProfileView = false
                            }
                        
                        
                        // showing the capsules for switching the pictures
                        if youProfileVM.userModel.userPhotos.count > 1 {
                            HStack(spacing: 10.0) {
                                ForEach(youProfileVM.userModel.userPhotos.sorted(by: <), id: \.key) { photoIndex, photoUrlString in
                                    IndicatorCapsule(tappedPhoto: $showPictureIndex, pictureIndex: photoIndex)
                                }
                            }
                            .offset(y: ((bounds.size.width - 48) * 1.33 / 2) - 45)
                        }
                        
                        
                        
                        // xmark symbol to show the user how to dismiss the view
                        HStack {
                            
                            Spacer()
                            
                            Image(systemName: "xmark")
                                .foregroundColor(Color("BackgroundSecondary").opacity(0.7))
                                .font(.system(size: 30))
                                
                                .onTapGesture(perform: {
                                    print("area button xmark - out of you profile")
                                    showYouProfileView = false
                            })
                            
                            Spacer()
                            
//                            // like button
//                            Button(action: {
//                                youProfileVM.addLikeToEvent(eventModel: event)
//                            }, label: {
//                                MatchButtonLabel(sfSymbol: "checkmark.circle.fill", color: Color.green)
//                            })
                        }
                        .offset(y: ((bounds.size.width - 48) * 1.33 / 2) + 60)
                        .frame(width: bounds.size.width - 48)
                        
                        
                    }
                    
                    
                    
                    
                    // spacer is used to get full area to tap
                    Spacer()
                    
                }
                
                
            }
            .frame(width: bounds.size.width, height: bounds.size.height, alignment: .center)
            .onAppear {
                youProfileVM.getYouProfil(eventModel: event)
            }
            
        }
        
    }
    
    
    // MARK: -
    var base: some View {
        
        // MARK: Base (including picture) of the Profile
        GeometryReader { bounds in
            ZStack {
                
                // MARK: White Background
                Color("Offwhite")
                    .frame(width: bounds.size.width - 48, height: (bounds.size.width - 48) * 1.33)
                    .clipShape(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                    )
                
                // MARK: Image downloaded from the Database
                URLImage(url: URL(string: youProfileVM.userModel.userPhotos[showPictureIndex] ?? stockUrlString)!) { image in
                    image.resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: bounds.size.width - 48 - 20, height: (bounds.size.width - 48) * 1.33 - 20)
                        .clipShape(
                            RoundedRectangle(cornerRadius: 20, style: .continuous)
                        )
                }
                
                // MARK: Tapping area for changing pictures
                HStack {
                    // left tap
                    Color.black.opacity(0.001)
                        .onTapGesture {
                            if showPictureIndex > 0 {
                                showPictureIndex -= 1
                            } else {
                                hapticFeedback(feedBackstyle: .error)
                            }
                            
                        }
                    
                    // right tap
                    Color.black.opacity(0.001)
                        .onTapGesture {
                            if showPictureIndex < youProfileVM.userModel.userPhotos.count - 1 {
                                showPictureIndex += 1
                            } else {
                                hapticFeedback(feedBackstyle: .error)
                            }
                            
                        }
                }
                .frame(width: bounds.size.width - 48, height: (bounds.size.width - 48) * 1.33)
                
            }
            .frame(width: bounds.size.width, height: bounds.size.height, alignment: .center)
        }
    }
    
    
    // MARK: -
    var informationBox: some View {
        
        // MARK: Information-Box at the bottom of the Profile
        GeometryReader { bounds in
            ZStack {
                
                // MARK: Background of the Information-Box
                BlurView(style: .systemUltraThinMaterial)
                    .frame(width: bounds.size.width - 48 - 26, height: 47)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(
                                LinearGradient(
                                    gradient: Gradient(stops: [
                                                        .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                                        .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                                    startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                                    endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                                lineWidth: 0.5
                            )
                    )
                    .clipShape(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                    )
                
                // MARK: Content of Information-Box
                HStack {
                    // name of the YOUser
                    Text(youProfileVM.userModel.name)
                    
                    // Pushing the texts to the ends
                    Spacer()
                    
                    // age of the YOUser
                    Text(String(dateToAge(date: youProfileVM.userModel.birthdayDate)))
                }
                .font(.largeTitle)
                .padding(.horizontal, 16)
                .frame(width: 301, height: 47)
            }
            .frame(width: bounds.size.width, height: bounds.size.height, alignment: .center)
        }
    }
    
    
    // MARK: -
    var eventCircle: some View {
        
        // MARK: Circle which shows the event picture
        ZStack {
            
            // Background Blur
            BlurView(style: .systemUltraThinMaterial)
                .frame(width: 91, height: 91)
                
                // Stroke to get glass effect
                .overlay(
                    Circle()
                        .stroke(
                            LinearGradient(
                                gradient: Gradient(stops: [
                                                    .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                                    .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                                startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                                endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                            lineWidth: 0.5
                        )
                )
                .clipShape(Circle())
            
            // Actual event image
            URLImage(url: URL(string: event.pictureURL)!) { image in
                image.resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 70, height: 70)
                    .clipShape(Circle())
            }
            
        }
    }
    
}

struct YouProfileNView_Previews: PreviewProvider {
    static var previews: some View {
        YouProfileNView(showYouProfileView: .constant(true), event: .constant(stockEvent))
    }
}
