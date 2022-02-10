//
//  MainChatView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 08.02.21.
//

import SwiftUI

struct MainChatView: View {
    
    @StateObject var firstActions: FirstActions = FirstActions()
    
    @State var showInformation: Bool = true
    
    var body: some View {
        
        
        ZStack {
            
            // MARK: List with chats available to the user
            ChatListView()
                .opacity(firstActions.firstViews["FirstChatArea"] == false ? 0.1 : firstActions.firstViews["FirstChatArea"] == nil ? 0.1 : 1)
            
            // MARK: InformationCard which is shown on first use
            if firstActions.firstViews["FirstChatArea"] == false || firstActions.firstViews["FirstChatArea"] == nil {
                InformationCard(goToNextView: $showInformation, goToLastView: $showInformation, sliderArray: [InformationCardModel(headerText: "Chat Area", highlight: true, footerText: "", image: "chat", sfSymbol: "bubble.left.and.bubble.right.fill", buttonText: "OK!", subtext: "Here you can chat with the people you matched. You can dissolve a match or delete the whole event with a long-press on each chat.")])
                    .onChange(of: showInformation, perform: { value in
                        // update variable to true that user does not see the info again
                        firstActions.firstViews["FirstChatArea"] = true
                        // save the status to the phone
                        firstActions.save()
                    })
            }
            
        }
    }
}

struct MainChatView_Previews: PreviewProvider {
    static var previews: some View {
        TabView {
            MainChatView()
        }
    }
}
