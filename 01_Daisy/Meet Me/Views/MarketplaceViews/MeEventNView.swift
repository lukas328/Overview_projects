//
//  MeEventNView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 08.03.21.
//

import SwiftUI
import URLImage

struct MeEventNView: View {
    
    @Binding var event: EventModel
    
    // represents the eventStatus of a MeEvent
    @State var eventStatus: EventStatus = .notLiked
    
    // to configure the date which is showing in the second line of the row
    var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.dateFormat = "d. MMM"
        return formatter
    }()
    var timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
    
    var body: some View {
        
        ZStack {
            
            // MARK: - Base (including picture) of the Event
            ZStack {
                
                // MARK: White Background
                Color("Offwhite")
                    .frame(width: 155, height: 155)
                    .clipShape(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                    )
                    .shadow(color: Color.black.opacity(0.25), radius: 4, x: 0, y: 4)
                
                // MARK: Image downloaded from the Database
                URLImage(url: URL(string: event.pictureURL) ?? stockURL) { image in
                    image.resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 145, height: 145)
                        .clipShape(
                            RoundedRectangle(cornerRadius: 23, style: .continuous)
                        )
                    
                }
            }
            
            
            
            // MARK: Information-Box at the bottom of the Event
            ZStack {
                
                // MARK: Background of the Information-Box
                BlurView(style: .systemUltraThinMaterial)
                    .frame(width: 157, height: 38)
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
                    // Category at the beginning
                    Text(event.category)
                    // Divider to separate Category and Time-data
                    Divider()
                        .padding(.horizontal, 8)
                    // Time-data in vertical order
                    VStack(spacing: 0.0) {
                        Text(dateFormatter.string(from: event.date))
                        Text(timeFormatter.string(from: event.startTime))
                            .foregroundColor(.accentColor)
                    }
                    .font(.subheadline)
                }
                .frame(width: 157, height: 35.0)
            }
            .offset(x: 10.0, y: 60)
            
            
            // MARK: EventStatus indicator, only showing if event is liked or matched
            if eventStatus != .notLiked {
                ZStack {
                    
                    // MARK: Background behind the Symbol
                    BlurView(style: .systemThinMaterial)
                        .frame(width: 36, height: 36)
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
                        .clipShape(
                            Circle()
                        )
                    
                    // MARK: Symbol which shows the eventStatus
                    Image(systemName: eventStatus == .liked ? "rosette" : eventStatus == .matched ? "lock" : "")
                        .font(.system(size: 24))
                        .foregroundColor(.accentColor)
                }
                .offset(x: 70, y: -70)
            }
            
        }
        // padding to show the shadow
        .padding(.top, 10)
        .padding(.bottom, 14)
        
        // MARK: If the status changes, change the indicator
        .onChange(of: event.likedUser, perform: { value in
            checkEventStatus()
        })
        .onChange(of: event.eventMatched, perform: { value in
            checkEventStatus()
        })
        // MARK: Initial status check
        .onAppear {
            checkEventStatus()
        }
        
        
        
    }
    
    // MARK: - Functions
    
    // MARK: Function to change the eventStatus
    func checkEventStatus() {
        
        // Check if the event is liked, not liked oder matched to show the right indicator
        if event.likedUser && !event.eventMatched {
            eventStatus = .liked
        } else if event.eventMatched {
            eventStatus = .matched
        }
    }
}

struct MeEventN_Previews: PreviewProvider {
    static var previews: some View {
        MeEventNView(event: .constant(stockEvent))
    }
}
