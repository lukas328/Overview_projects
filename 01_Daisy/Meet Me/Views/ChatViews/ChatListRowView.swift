//
//  ChatListRowView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.02.21.
//

import SwiftUI
import URLImage

struct ChatListRowView: View {
    
    @Binding var match: AllMatchInformationModel
    
    @Binding var chatTapped: Bool
    @Binding var matchTapped: AllMatchInformationModel
    @Binding var matchLongPressed: AllMatchInformationModel
    
    @State var firstPartString: String = ""
    
    // to configure the date which is showing in the second line of the row
    var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.dateFormat = "EEEE, d MMM"
        return formatter
    }()
    var timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
    
    var body: some View {
        
        // MARK: RowView of one line in the list
        ZStack {
            HStack {
                
                // profile picture
                URLImage(url: URL(string: match.user.userPhotos[0] ?? stockUrlString)!) { image in
                    image.resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: 60)
                        .overlay(
                            Circle()
                                .stroke(
                                    LinearGradient(gradient: Gradient(colors: [Color.white.opacity(0.9), Color.gray]), startPoint: .topTrailing, endPoint: .bottomLeading),
                                    lineWidth: 4
                                )
                        )
                        .clipShape(Circle())
                }
                
                VStack(alignment: .leading) {
                    
                    // first row with event-type
                    HStack(spacing: 0.0) {
                        Text(firstPartString)
                        Text(" \(match.user.name)")
                            .foregroundColor(.accentColor)
                    }
                    .lineLimit(1)
                    
                    // second row with event date
                    HStack(spacing: 0.0) {
                        Text(dateFormatter.string(from: match.event.date))
                            
                        Text(" at ")
                            
                        Text(timeFormatter.string(from: match.event.startTime))
                            
                    }
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    
                }
                .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(8)
            .modifier(offWhiteShadow(cornerRadius: 12))
            .padding(.horizontal, 10)
            
            // show soon when Event is in the next 12 hours and "event over when it's over
            if Date().distance(to: combineDateWithTime(date: match.event.date, time: match.event.startTime)!) < 43200 {
                HStack {
                    Text(Date().distance(to: combineDateWithTime(date: match.event.date, time: match.event.startTime)!) < 0 ? "Event over" : "Soon!")
                        .font(.system(size: 14))
                        .padding(5)
                        .foregroundColor(.white)
                        .background(Color.accentColor)
                        .mask(Capsule())
                        .shadow(color: Color.black.opacity(0.3), radius: 4, x: 2, y: 2)
                        .padding(.leading, 80)
                    Spacer()
                    
                }
                .offset(y: -35)
            }
            
            // show a marker when a new message is in the chat
            if match.unReadMessage {
                HStack {
                    Spacer()
                    Text("!")
                        .font(.system(size: 14))
                        .padding(10)
                        .foregroundColor(.white)
                        .background(Color.accentColor)
                        .mask(Circle())
                        .shadow(color: Color.black.opacity(0.3), radius: 4, x: 2, y: 2)
                        .padding(.trailing, 40)
                }
                .offset(y: -35)
            }
            
        }
        .padding(.vertical, 10)
        // tapping to show the chat
        .onTapGesture {
            matchTapped = match
            chatTapped = true
        }
        
        // long-pressing to delete the event
        .onLongPressGesture {
            matchLongPressed = match
        }
        
        // on appear
        .onAppear {
            switch match.event.category {
            
            // event for meeting for a walk
            case "Walk":
                firstPartString = "Go for a walk with"
            // event for meeting in a cafe
            case "Café":
                firstPartString = "Drinking coffee with"
            // event for meeting eating together
            case "Food":
                firstPartString = "Eating with"
            // event for meeting for doing sports together
            case "Sport":
                firstPartString = "Doing an exercise with"
            // event for meeting in a bar
            case "Bar":
                firstPartString = "Having drinks with"
            // event for meeting for everything else which is not listed above
            default:
                firstPartString = "\(match.event.category) with"
            }
        }
    }
    
    // MARK: Functions
    
    // combine the day form the date and time from startTimeDate
    func combineDateWithTime(date: Date, time: Date) -> Date? {
        let calendar = NSCalendar.current
        
        let dateComponents = calendar.dateComponents([.year, .month, .day], from: date)
        let timeComponents = calendar.dateComponents([.hour, .minute, .second], from: time)
        
        var mergedComponents = DateComponents()
        mergedComponents.year = dateComponents.year!
        mergedComponents.month = dateComponents.month!
        mergedComponents.day = dateComponents.day!
        mergedComponents.hour = timeComponents.hour!
        mergedComponents.minute = timeComponents.minute!
        mergedComponents.second = timeComponents.second!
        
        return calendar.date(from: mergedComponents)
    }
}

struct ChatListRowView_Previews: PreviewProvider {
    static var previews: some View {
        ChatListRowView(match: .constant(AllMatchInformationModel(chatId: "", unReadMessage: false, user: stockUser, event: stockEvent)), chatTapped: .constant(false), matchTapped: .constant(AllMatchInformationModel(chatId: "", unReadMessage: false, user: stockUser, event: stockEvent)), matchLongPressed: .constant(AllMatchInformationModel(chatId: "", unReadMessage: false, user: stockUser, event: stockEvent)))
    }
}
