//
//  MeEventLineView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 17.02.21.
//

import SwiftUI
import PromiseKit

struct MeEventLineView: View {
    
    @StateObject private var meEventLineVM = MeEventLineViewModel()
    
    
    @State var buttonPressed: Bool = true
    
    @Binding var showCreationView: Bool
    @Binding var showMeMatchView: Bool
    
    @Binding var tappedEvent: EventModel
    
    let notchPhone: Bool = UIApplication.shared.windows[0].safeAreaInsets.bottom > 0 ? true : false
    
    var body: some View {
        GeometryReader { geometry in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 20.0) {
                    
                    // MARK: Button to add new event
                    Image(systemName: "plus.circle")
                        .font(.title)
                        .padding(10)
                        .padding(.vertical, 40)
                        .background(BlurView(style: .systemMaterial))
                        .clipShape(Circle())
                        .shadow(color: Color.black.opacity(0.1), radius: 1, x: 0, y: 1)
                        .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: 10)
                        .padding(.leading, 35)
                        .padding(.trailing, 10)
                        .scaleEffect(buttonPressed ? 0.8 : 1)
                        .opacity(buttonPressed ? 0.5 : 1)
                        .animation(.spring(response: 0.3, dampingFraction: 0.3, blendDuration: 0.3))
                        .padding()
                        .onTapGesture {
                            
                            // button animation start
                            buttonPressed.toggle()
                            
                            // haptic feedback when button is tapped
                            hapticPulse(feedback: .rigid)
                            
                            // close view
                            showCreationView = true
                        }
                    
                    // show message if no events are available
                    if meEventLineVM.eventArray.count < 1 {
                        Text("Create your own event, now. Just hit the + button")
                            .padding(8)
                            .frame(width: 130, height: 130, alignment: .center)
                            .background(Color("Offwhite"))
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .shadow(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 2)
                            .padding(.leading, 10)
                    }
                    
                    // MARK: List with own events
                    ForEach(meEventLineVM.eventArray.indices, id: \.self) { event in
                        HStack {
                            MeEventNView(event: $meEventLineVM.eventArray[event])
                                .scaleEffect(notchPhone ? 1 : 0.8)
                                .rotation3DEffect(
                                    // get new angle, move the min x 30pt more to the right and make the whole angle smaller with the / - 40
                                    Angle(
                                        degrees: Double(geometry.frame(in: .global).minX - 30) / -20),
                                    axis: (x: 0, y: 10, z: 0)
                                )
                                
                                // tap on each of the events does that
                                .onTapGesture {
                                    
                                    // save UserModelObject
                                    tappedEvent = meEventLineVM.eventArray[event]
                                    
                                    // shows the MeMatch
                                    showMeMatchView = true
                                }
                        }
                    }
                }
            }
            .offset(y: notchPhone ? 0 : -10)
        }        
        .onAppear {
            self.meEventLineVM.getMeEvents()
        }
    }
}

struct MeEventLineView_Previews: PreviewProvider {
    static var previews: some View {
        MeEventLineView(showCreationView: .constant(false), showMeMatchView: .constant(false), tappedEvent: .constant(stockEvent))
    }
}
