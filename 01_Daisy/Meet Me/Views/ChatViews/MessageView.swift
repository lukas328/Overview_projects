//
//  MessageView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.02.21.
//

import SwiftUI
import Firebase

enum MessageStyle {
    case creator
    case receiver
    case noUser
}

struct MessageView: View {
    
    @StateObject var messagesVM: MessagesViewModel = MessagesViewModel()
    @Binding var message: MessageModel
    @State var messageStyle: MessageStyle = .receiver
    
    var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_DE")
        formatter.dateFormat = "HH:mm - EEEE"
        return formatter
    }()
    
    var body: some View {
        VStack {
            VStack(alignment: messageStyle == .creator ? .trailing : .leading, spacing: 5) {
                Text(dateFormatter.string(from: message.timeStamp.dateValue()))
                    .font(.footnote)
                    .foregroundColor(messageStyle == .creator ? .white : .accentColor)
                
                Text(message.messageText)
            }
            .padding(8)
            .background(messageStyle == .receiver ? Color.gray : messageStyle == .noUser ? Color("BackgroundSecondary") : Color.accentColor)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .frame(width: 250, alignment: messageStyle == .creator ? .trailing : .leading)
            
            
            
        }
        .frame(maxWidth: .infinity, alignment: messageStyle == .creator ? .trailing : .leading)
        .padding()
        // on appear to change MessageStyle
        .onAppear {
            print(message.userId)
            // check if the message was received or sent
            switch message.userId {
            case "NoUser":
                messageStyle = .noUser
            case "":
                messageStyle = .noUser
            case messagesVM.userId:
                messageStyle = .creator
            default:
                messageStyle = .receiver
            }
            
        }
    }
}

struct MessageView_Previews: PreviewProvider {
    static var previews: some View {
        MessageView(message: .constant(stockChat.messages[1]))
    }
}
