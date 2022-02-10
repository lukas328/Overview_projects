//
//  MeEventControllerView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 21.02.21.
//

import SwiftUI
import PromiseKit

struct MeEventControllerView: View {
    
    @StateObject private var meEventControllerVM: MeEventControllerViewModel = MeEventControllerViewModel()
    @ObservedObject var firstActions: FirstActions = FirstActions()
    
    @Binding var showMeEventControllerView: Bool
    @Binding var tappedEvent: EventModel
    
    @State var userChosen: Bool = false
    @State var showMeMatchCardView: Bool = false
    @State var showInformation: Bool = true
    @State var showFirstMatchInfo: Bool = true
    
    var body: some View {
        
        ZStack {
            
            
            
            // MARK: Stack with Users to swipe and like
            ZStack {
                ForEach(meEventControllerVM.likedUsers.indices, id: \.self ) { userNumber in
                    MeMatchNCardView(showMeMatchNCardView: $showMeMatchCardView, showMeEventControllerView: .constant(true), likedUsers: $meEventControllerVM.likedUsers, userAccepted: $userChosen, event: tappedEvent, userIndex: userNumber)
                        .opacity(showMeMatchCardView ? 1 : 0)
                        .onChange(of: userChosen) { (chosen) in
                            
                            if firstActions.firstViews["FirstMatch"] == false || firstActions.firstViews["FirstMatch"] == nil {
                                showMeEventControllerView = true
                            } else {
                                showMeEventControllerView = false
                            }
                        }
                        
                }
            }
            
            // MARK: First swipe information
            if showMeMatchCardView && (firstActions.firstViews["FirstSwipe"] == false || firstActions.firstViews["FirstSwipe"] == nil) {
                InformationCard(goToNextView: $showInformation, goToLastView: $showInformation, sliderArray: [InformationCardModel(headerText: "Match user", highlight: true, footerText: "", image: "swipe-match", sfSymbol: "heart.fill", buttonText: "OK!", subtext: "You can choose between all users who liked your event. Once you accepted someone - you both meet, so choose wisely! You never know if it's the last one you see...")])
                    .onChange(of: showInformation, perform: { value in
                        // update variable to true that user does not see the info again
                        firstActions.firstViews["FirstSwipe"] = true
                        // save the status to the phone
                        firstActions.save()
                    })
            }
            
            // MARK: First match information
            if userChosen && (firstActions.firstViews["FirstMatch"] == false || firstActions.firstViews["FirstMatch"] == nil) {
                InformationCard(goToNextView: $showFirstMatchInfo, goToLastView: $showFirstMatchInfo, sliderArray: [InformationCardModel(headerText: "User matched", highlight: true, footerText: "", image: "swipe-match", sfSymbol: "heart.fill", buttonText: "OK!", subtext: "You have got your first match. Go to the chat area to discuss further details")])
                    .onChange(of: showFirstMatchInfo, perform: { showInfo in
                        // update variable to true that user does not see the info again
                        print("onchange card first match")
                        firstActions.firstViews["FirstMatch"] = true
                        // save the status to the phone
                        firstActions.save()
                        
                        // close the view
                        showMeEventControllerView = false
                    })
                    .onAppear {
                        print("appeared")
                    }
            }
            
            // MARK: DetailedView to start matching or editing/deleting the event
            MeEventDetailedView(showMeEventDetailedView: $showMeEventControllerView, showMeMatchCardView: $showMeMatchCardView, event: $tappedEvent, likedUsers: $meEventControllerVM.likedUsers)
                .opacity(showMeMatchCardView ? 0 : 1)
                .opacity(userChosen ? 0 : 1)
            
        }
        .animation(.easeInOut)
        
        // MARK: OnAppear to get the likedUsers from the Database
        .onAppear {
            meEventControllerVM.getLikedUsers(eventId: tappedEvent.eventId)
            
        }
    }
}

struct MeMatchView_Previews: PreviewProvider {
    static var previews: some View {
        MeEventControllerView(showMeEventControllerView: .constant(true), tappedEvent: .constant(stockEvent))
    }
}
