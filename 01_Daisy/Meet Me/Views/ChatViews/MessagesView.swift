//
//  MessagesView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.02.21.
//

import SwiftUI
import URLImage

struct MessagesView: View {
    
    @StateObject var messagesVM : MessagesViewModel = MessagesViewModel()
    @Binding var match: AllMatchInformationModel
    
    // message which needs to be uploaded
    @State var newMessage: String = ""
    
    @State var firstPartString: String = ""
    @State var showChatProfileView: Bool = false
    @State var showChatEventView: Bool = false
    
    @State var eventWithOtherUser: EventModel?
    
    let notchPhone: Bool = UIApplication.shared.windows[0].safeAreaInsets.bottom > 0 ? true : false
    
    var body: some View {
        
        ZStack {
            
            VStack {
                
                // MARK: Message area
                ZStack {
                    
                    
                    ScrollView {
                        VStack {
                            ForEach(messagesVM.chat.messages.indices.reversed(), id: \.self) { messageNumber in
                                MessageView(message: $messagesVM.chat.messages[messageNumber])
                                    .rotationEffect(.radians(.pi))
                                    .scaleEffect(x: -1, y: 1, anchor: .center)
                                    .onTapGesture {
                                        UIApplication.shared.endEditing()
                                    }
                            }
                        }
                        .background(
                            Color.white.opacity(0.001)
                                .onTapGesture {
                                    // dismiss keyboard when tapped the background
                                    UIApplication.shared.endEditing()
                                }
                        )
                        
                    }
                    .rotationEffect(.radians(.pi))
                    .scaleEffect(x: -1, y: 1, anchor: .center)
                    
                    // MARK: Top area
                    VStack {
                        topBar
                        Spacer()
                    }
                }
                
                
                // MARK: Send and type area
                HStack {
                    TextField("Type new message here...", text: $newMessage)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .padding(.leading, 5)
                    
                    Button(action: {
                        
                        messagesVM.UploadChat(allMatchInformation: match, messageText: newMessage)
                        newMessage = ""
                        
                    }, label: {
                        Image(systemName: "paperplane.fill")
                            .padding()
                    })
                }
                .padding(8)
                .modifier(offWhiteShadow(cornerRadius: 12))
                .padding(.horizontal, 16)
                .padding(.bottom, 18)
            }
            .opacity(showChatProfileView ? 0.1 : 1)
            .opacity(showChatEventView ? 0.1 : 1)
            .onAppear {
                eventWithOtherUser = match.event
                eventWithOtherUser!.userId = match.user.userId
            }

            if showChatProfileView {
                YouProfileNView(showYouProfileView: $showChatProfileView, event: Binding($eventWithOtherUser)!)
                    .scaleEffect(notchPhone ? 1 : 0.85)
            }
            
            
            if showChatEventView {
                ChatEventView(event: $match.event, showChatEventView: $showChatEventView)
                    .padding(.trailing, 20)
            }
            
        }
        .background(
            ZStack {
                Image("background")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            }
        )
        .navigationBarTitleDisplayMode(.inline)
        
        
        // MARK: - Top area (inside the toolbar)
        .toolbar {
            // MARK: 
            ToolbarItem {
                HStack(spacing: 0.0) {
                    Text(firstPartString)
                    Text(" ")
                    Text(match.user.name)
                }
                .foregroundColor(.primary)
            }
            
        }
        
        .onAppear {
            
            messagesVM.unReadMessageFalse(chatId: match.chatId)
            messagesVM.downloadChat(chatId: match.chatId)
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
    
    
    var topBar: some View {
        HStack {
            
            // MARK: Show the event picture
            URLImage(url: URL(string: match.event.pictureURL) ?? stockURL) { image in
                    image.resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 80, height: 80)
                    .overlay(
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .stroke(
                                LinearGradient(gradient: Gradient(colors: [Color.white.opacity(0.9), Color.gray]), startPoint: .topTrailing, endPoint: .bottomLeading),
                                lineWidth: 4
                            )
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                    .padding(.trailing, 20)
            }
            .onTapGesture {
                withAnimation(.default) {
                    UIApplication.shared.endEditing() 
                    showChatEventView.toggle()
                }
            }
                
            // MARK: Show the profile picture
            URLImage(url: URL(string: match.user.userPhotos[0] ?? stockUrlString)!) { image in
                    image.resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 80, height: 80)
                    .overlay(
                        Circle()
                            .stroke(
                                LinearGradient(gradient: Gradient(colors: [Color.white.opacity(0.9), Color.gray]), startPoint: .topTrailing, endPoint: .bottomLeading),
                                lineWidth: 4
                            )
                    )
            .clipShape(Circle())
            }
            .onTapGesture {
                withAnimation(.default) {
                    UIApplication.shared.endEditing()
                    showChatProfileView.toggle()
                }
            }
            
        }
        .padding(8)
        
        // MARK: Background of the TopBar
        .background(
            BlurView(style: .systemUltraThinMaterial)
                .overlay(
                    HalfRoundedRectangle()
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
                    HalfRoundedRectangle()
                )
        )
    }
}

struct HalfRoundedRectangle: Shape {
    
    func path(in rect: CGRect) -> Path {
        
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: [.bottomLeft, .bottomRight], cornerRadii: CGSize(width: 13, height: 13))
        
        return Path(path.cgPath)
    }
}


struct MessagesView_Previews: PreviewProvider {
    static var previews: some View {
        MessagesView(match: .constant(AllMatchInformationModel(chatId: "08470AAA-128F-46A3-9D23-1CD48C528938", unReadMessage: false, user: stockUser, event: stockEvent)))
    }
}
