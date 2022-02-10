//
//  YouEventLineView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 10.02.21.
//

import SwiftUI
import PromiseKit
import MapKit


struct YouEventLineView: View {
    
    @ObservedObject private var showedEventsModel = ShowedEventsModel()
    @ObservedObject private var firstActions = FirstActions()
    
    @StateObject private var youEventLineVM = YouEventLineViewModel()
    @EnvironmentObject var locationManager: LocationManager
    // data transfer form database
    @State var eventArray: [EventModel] = []
    @State private var loading: Bool = false
    
    @Binding var tappedYouEvent: EventModel
    @Binding var showYouProfileView: Bool
    @Binding var showSuccess: Bool
    
    // index which event needs to be removed from the array/scrollview
    @State var eventRemoveIndex: Int = -1
    @GestureState var longPress: Bool = false
    @State var pressDone: Bool = false
    
    let notchPhone: Bool = UIApplication.shared.windows[0].safeAreaInsets.bottom > 0 ? true : false
    
    var body: some View {
        ZStack {
            
            // MARK: Dashed rectangle for dragging
            if eventArray.count >= 1 {
                Text("Drag Events down you like to join")
                    .frame(width: 300, height: 25, alignment: .center)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(style: StrokeStyle(
                                lineWidth: 2,
                                dash: [10]
                            ))
                            .gradientForeground(gradient: secondaryGradient)
                    )
                    .offset(y: notchPhone ? 150 : 120)
            }
            
            
            // MARK: Horizontal event list
            VStack {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 30) {
                        
                        // show message if no events are available
                        if eventArray.count < 1 {
                            Text("Sorry, we couldn't find any events with your filter, because we are pretty new. Change your settings and REFRESH.")
                                .padding(8)
                                .frame(width: 175, height: 175, alignment: .center)
                                .modifier(offWhiteShadow(cornerRadius: 12))
                                .padding(.leading, 10)
                                .frame(width: 250, height: 250)
                                .padding(.top, 30)
                                .padding(.bottom, 120)
                            
                        }
                        
                        // create a view for each event in the array
                        ForEach(eventArray.indices, id: \.self) { eventIndex in
                            GeometryReader { geometry in
                                
                                YouEventNView(events: $eventArray, showSuccess: $showSuccess, eventIndex: eventIndex, currentEvent: eventArray[eventIndex])
                                    .scaleEffect(notchPhone ? 1 : 0.7)
                                    .rotation3DEffect(
                                        // get new angle, move the min x 30pt more to the right and make the whole angle smaller with the / - 40
                                        Angle(
                                            degrees: Double(geometry.frame(in: .global).minX - 30) / -40),
                                        axis: (x: 0, y: 10, z: 0)
                                    )
                                    
                                    .onTapGesture {
                                        tappedYouEvent = eventArray[eventIndex]
                                        withAnimation(.easeIn(duration: 0.1)) {
                                            showYouProfileView = true
                                        }
                                    }
                                    .contextMenu(ContextMenu(menuItems: {
                                        Button(action: {
                                            
                                            // TODO: @budni add the backend content here, as you see you can delete from eventArray without problems
                                            youEventLineVM.reportEvent(eventModel: eventArray[eventIndex])
                                            
                                            eventArray.remove(at: eventIndex)
                                            showedEventsModel.events = eventArray
                                            showedEventsModel.save()
                                            
                                            
                                        }, label: {
                                            Label("Report and dismiss event", systemImage: "exclamationmark.circle")
                                        })
                                        
                                    }))
                            }
                            
                            .frame(width: notchPhone ? 250 : 200, height: notchPhone ? 250 : 200)
                            .padding(.bottom, notchPhone ? 120 : 80)
                            .padding(.leading, 30)
                            .padding(.top, notchPhone ? 30 : 15)
                            .offset(y: notchPhone ? 0 : -30)
                        }
                        // needed to update dragged event and array
                        .id(UUID())
                        
                        // button at the end to refresh events
                        refreshButton
                            .frame(width: notchPhone ? 250 : 200, height: notchPhone ? 250 : 200)
                            .padding(.bottom, notchPhone ? 120 : 80)
                            .padding(.leading, eventArray.count < 1 ? -10 : 30)
                            .padding(.top, notchPhone ? 30 : 15)
                            .padding(.trailing, 20)
                        
                    }
                }
                .frame(maxWidth: .infinity)
                
                
                Spacer()
            }
            .frame(height: notchPhone ? 380 : 300)
            
            // loading screen deactivated
            //            LoadingView(showLoadingScreen: $loading)
            
            //                boomBar
            //                    .opacity(pressDone ? 0 : 0.95)
            
            
            
            
        }
        .frame(height: notchPhone ? 380 : 300)
        .onAppear {
            youEventLineVM.getRefreshCounter().done {
                if youEventLineVM.userModel.userStatus == "test" {
                    firstly {
                        self.youEventLineVM.getYouEventsTest(shuffle: true)
                    }.done { events in
                        self.eventArray = events
                        showedEventsModel.events = events
                        showedEventsModel.save()
                        //self.youEventLineVM.addOneToRefreshCounter()
                        
                        //add FirstventShuffle to Dictionary
                        firstActions.firstViews["FirstEventShuffle"] = true
                        firstActions.save()
                        print("aufgerufen")
                        print("done")
                    }.catch { error in
                        print("DEBUG: error in GetYouEventChain: \(error)")
                        print("DEBUG: \(error.localizedDescription)")
                    }
                    
                } else {
                    
                    //Check if it is the first login
                    if firstActions.firstViews["FirstEventShuffle"] == false || firstActions.firstViews["FirstEventShuffle"] == nil {
                        //if it is the first Login load events without hitting shuffle button
                        if youEventLineVM.refreshCounter < 10 || youEventLineVM.userModel.userStatus == "developer" {
                            
                            firstly {
                                self.youEventLineVM.getYouEvents(region: locationManager.region, shuffle: true)
                            }.done { events in
                                self.eventArray = events
                                showedEventsModel.events = events
                                showedEventsModel.save()
                                //self.youEventLineVM.addOneToRefreshCounter()
                                
                                //add FirstventShuffle to Dictionary
                                firstActions.firstViews["FirstEventShuffle"] = true
                                firstActions.save()
                                print("aufgerufen")
                                print("done")
                            }.catch { error in
                                print("DEBUG: error in GetYouEventChain: \(error)")
                                print("DEBUG: \(error.localizedDescription)")
                            }
                        }
                    }else {
                        print("aufgerufen: \(eventArray)")
                        self.eventArray = showedEventsModel.events
                    }
                }
            }.catch { error in
                print(error)
            }
            
        }
        
        
    }
    
    // MARK: Button to get new events
    var refreshButton: some View {
        
        
        ZStack {
            
            // actual button
            VStack(spacing: 12.0) {
                Text("Press and hold to")
                    .font(.footnote)
                
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 40))
                    .gradientForeground(gradient: secondaryGradient)
                
                Text("REFRESH")
            }
            .frame(width: 175, height: 175, alignment: .center)
            .modifier(offWhiteShadow(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .trim(from: longPress ? 0 : 1, to: 1)
                    .stroke(Color.secondary, style: StrokeStyle(lineWidth: longPress ? 30 : 4, lineCap: .round, lineJoin: .round, miterLimit: 0, dash: [longPress ? 800 : 0, 1000], dashPhase: 0))
            )
            .gesture(LongPressGesture(minimumDuration: 1)
                        .updating($longPress) { currentState, gestureState, transaction in
                            
                            transaction.animation = Animation.easeInOut(duration: 1.0)
                            gestureState = currentState
                            
                        }
                        .onEnded { value in
                            pressDone = true
                            hapticFeedback(feedBackstyle: .success)
                            if youEventLineVM.userModel.userStatus == "test" {
                                firstly {
                                    self.youEventLineVM.getYouEventsTest(shuffle: true)
                                }.done { events in
                                    self.eventArray = events
                                    showedEventsModel.events = events
                                    showedEventsModel.save()
                                    //self.youEventLineVM.addOneToRefreshCounter()
                                    
                                    //add FirstventShuffle to Dictionary
                                    firstActions.firstViews["FirstEventShuffle"] = true
                                    firstActions.save()
                                    print("aufgerufen")
                                    print("done")
                                }.catch { error in
                                    print("DEBUG: error in GetYouEventChain: \(error)")
                                    print("DEBUG: \(error.localizedDescription)")
                                }
                                
                            } else {
                            
                            if youEventLineVM.refreshCounter < 10 || youEventLineVM.userModel.userStatus == "developer" {
                                self.eventArray = []
                                showedEventsModel.events = eventArray
                                showedEventsModel.save()
                                // refresh youEvents
                                firstly {
                                    self.youEventLineVM.getYouEvents(region: locationManager.region, shuffle: true)
                                }.done { events in
                                    self.eventArray = events
                                    showedEventsModel.events = events
                                    showedEventsModel.save()
                                    
                                }.catch { error in
                                    print("DEBUG: error in GetYouEventChain: \(error)")
                                    print("DEBUG: \(error.localizedDescription)")
                                }
                                self.youEventLineVM.addOneToRefreshCounter()
                            }else {
                                //Feedback for User that maximum amount of refreshes are reached
                                
                            }
                            }
                            
                            
                        })
            .padding(30)
            .frame(width: 250, height: 250, alignment: .center)
            
            // overlay with number
            Text("\(youEventLineVM.refreshCounter)/10")
                .padding(8)
                .background(Color(UIColor(.accentColor)))
                .mask(Capsule())
                .offset(x: 175 / 2, y: -175 / 2)
        }
        
    }
    
    
    // MARK: BoomBar
    var boomBar: some View {
        Image("Beam")
            .resizable()
            .aspectRatio(contentMode: .fill)
            //            .frame(maxWidth: pressDone ? .infinity : 0, maxHeight: pressDone ? .infinity : 0)
            .frame(width: 400, height: 350, alignment: /*@START_MENU_TOKEN@*/.center/*@END_MENU_TOKEN@*/)
            .offset(x: pressDone ? 0 : -1500)
            .animation(.easeOut(duration: 1.5))
            .padding(.vertical, 5)
            .onChange(of: pressDone, perform: { value in
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5, execute: {
                    pressDone = false
                    
                })
            })
    }
    
}

struct EventLineView_Previews: PreviewProvider {
    static var previews: some View {
        YouEventLineView(tappedYouEvent: .constant(stockEvent), showYouProfileView: .constant(true), showSuccess: .constant(false))
    }
}
