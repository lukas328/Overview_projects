//
//  MeEventDetailedView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 13.03.21.
//

import SwiftUI
import URLImage

struct MeEventDetailedView: View {
    
    @StateObject var changeMeEventVM: ChangeMeEventViewModel = ChangeMeEventViewModel()
    
    @Binding var showMeEventDetailedView: Bool
    
    @Binding var showMeMatchCardView: Bool
    @Binding var event: EventModel
    @Binding var likedUsers: [UserModel]
    
    @State var eventStatus: EventStatus = .liked
    
    var body: some View {
        
        GeometryReader { bounds in
            ZStack {
                
                Color.black.opacity(0.001)
                    .onTapGesture(perform: {
                        print("dismiss")
                        showMeEventDetailedView = false
                        
                    })
                
                VStack {
                    
                    // spacer is used to get full area to tap
                    Spacer()
                    
                    
                    ZStack {
                        
                        // base including the vector stripes
                        base
                        
                        // show the event at the top right corner
                        eventCircle
                            .offset(x: (bounds.size.width - 48) / 2 - 30, y: -((bounds.size.width - 48) * 1.33) / 2 + 30)
                        
                        // text to inform user about the current MeEvent Status
                        FrozenTextBox(text: eventStatus == .matched ? "You already have someone to meet, go to the chat area to set up the details!" : eventStatus == .liked ? "Ready to find someone for your event? Tap on start match." : "Wait a bit that others can find your Event...")
                        
                        // ---
                        
                        // button to delete the current event
                        Button(action: {
                            // deleting Me Event from Database
                            changeMeEventVM.deleteMeEvent(eventModel: event)
                            showMeEventDetailedView = false
                            
                        }, label: {
                            FrozenButton(text: "Delete Event", sfSymbol: "pencil.circle")
                        })
                        .offset(y: 150)
                        
                        // button to start the matching / swipe
                        Button(action: {
                            // TODO: add action when user taps start match
                            
                            // show me match view now
                            if likedUsers.count != 0 {
                                showMeMatchCardView = true
                            } else {
                                showMeEventDetailedView = false
                            }
                            
                            // haptic feedback when button is tapped
                            hapticPulse(feedback: .rigid)
                            
                        }, label: {
                            FrozenButton(text: "Start Match", sfSymbol: "")
                                .scaleEffect(eventStatus == .liked ? 1.2 : 1)
                        })
                        .offset(y: ((bounds.size.width - 48) * 1.33 / 2) - 10)
                        .offset(y: eventStatus == .liked ? 5 : 0)
                        .disabled(eventStatus == .matched ? true : eventStatus == .notLiked ? true : false)
                        .opacity(eventStatus == .matched ? 0.7 : eventStatus == .notLiked ? 0.7 : 1)
                        
                        // ---
                        
                        // xmark symbol to show the user how to dismiss the view
                        Image(systemName: "xmark")
                            .foregroundColor(Color("BackgroundSecondary").opacity(0.5))
                            .font(.system(size: 30))
                            .offset(y: ((bounds.size.width - 48) * 1.33 / 2) + 60)
                    }
                    
                    // spacer is used to get full area to tap
                    Spacer()
                    
                }
                
            }
            .frame(width: bounds.size.width, height: bounds.size.height, alignment: .center)
            
            // MARK: OnAppear to get the event Status
            .onAppear {
                
                // get the event status
                if event.eventMatched {
                    eventStatus = .matched
                } else if event.likedUser {
                    eventStatus = .liked
                } else {
                    eventStatus = .notLiked
                }
                
            }
            
        }
    }
    
    
    // MARK: -
    var base: some View {
        
        // MARK: Base of the Me Event detail view
        GeometryReader { bounds in
            ZStack {
                
                // MARK: White Background
                Color("Offwhite")
                    .frame(width: bounds.size.width - 48, height: (bounds.size.width - 48) * 1.33)
                    .clipShape(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                    )
                
                
                // MARK: Background vector
                Image("me-event-background")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .scaleEffect(1.1)
                    .frame(width: bounds.size.width - 48, height: (bounds.size.width - 48) * 1.33)
                    .clipShape(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                    )
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
            URLImage(url: URL(string: event.pictureURL) ?? stockURL) { image in
                image.resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 70, height: 70)
                    .clipShape(Circle())
            }
            
        }
    }
    
}

// MARK: - FrozenTextBox
struct FrozenTextBox: View {
    
    var text: String = ""
    
    var body: some View {
        
        // MARK: Text-box in the middle of the screen
        ZStack {
            
            // MARK: Background of the Text-Box
            BlurView(style: .systemUltraThinMaterial)
                .frame(width: 232, height: 228)
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
            
            // MARK: Content of text-Box
            Text(text)
                .font(.title3)
                .padding(30)
                .frame(width: 232, height: 228)
        }
    }
}


// MARK: - FrozenButton
struct FrozenButton: View {
    
    var text: String
    var sfSymbol: String
    
    var body: some View {
        Label(
            title: { Text(text) },
            
            icon: {
                if sfSymbol != "" {
                    Image(systemName: sfSymbol)
                }
            }
        )
        .frame(width: 138, height: 47)
        .background(
            BlurView(style: .systemUltraThinMaterial)
                .frame(width: 138, height: 47)
                
                // Stroke to get glass effect
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
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        )
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
